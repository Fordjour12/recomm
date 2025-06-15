package main

import (
	"fmt"
	"math"
	"sort"
)

// User represents a founder or investor
type User struct {
	ID       string
	Role     string   // "founder" or "investor"
	Industry []string // e.g., ["fintech", "AI"]
	Stage    string   // e.g., "seed", "series-a"
}

// Startup represents a startup
type Startup struct {
	ID        string
	FounderID string
	Industry  []string
	Stage     string
}

// Interaction represents user actions (e.g., investment, like)
type Interaction struct {
	UserID   string
	TargetID string // Startup or User ID
	Type     string // e.g., "invested", "liked"
	Score    float64
}

// Recommendation holds a recommended item
type Recommendation struct {
	TargetID string
	Type     string // "startup" or "user"
	Score    float64
}

// DataStore holds in-memory data (replace with database in production)
type DataStore struct {
	Users        map[string]User
	Startups     map[string]Startup
	Interactions map[string]map[string]float64 // UserID -> TargetID -> Score
	Features     map[string]map[string]float64 // ItemID -> Feature -> Weight
}

// NewDataStore initializes the data store
func NewDataStore() *DataStore {
	return &DataStore{
		Users:        make(map[string]User),
		Startups:     make(map[string]Startup),
		Interactions: make(map[string]map[string]float64),
		Features:     make(map[string]map[string]float64),
	}
}

// Add sample data for testing
func (ds *DataStore) LoadSampleData() {
	// Users
	ds.Users["u1"] = User{ID: "u1", Role: "founder", Industry: []string{"fintech"}, Stage: "seed"}
	ds.Users["u2"] = User{ID: "u2", Role: "investor", Industry: []string{"fintech", "AI"}, Stage: "seed"}
	ds.Users["u3"] = User{ID: "u3", Role: "investor", Industry: []string{"AI"}, Stage: "series-a"}

	// Startups
	ds.Startups["s1"] = Startup{ID: "s1", FounderID: "u1", Industry: []string{"fintech"}, Stage: "seed"}
	ds.Startups["s2"] = Startup{ID: "s2", FounderID: "u1", Industry: []string{"AI"}, Stage: "series-a"}

	// Interactions
	ds.Interactions["u1"] = map[string]float64{"s2": 1.0}            // Founder liked AI startup
	ds.Interactions["u2"] = map[string]float64{"s1": 5.0, "u1": 2.0} // Investor funded fintech startup, followed founder
	ds.Interactions["u3"] = map[string]float64{"s2": 5.0}            // Investor funded AI startup

	// Features (simplified TF-IDF-like weights)
	ds.Features["s1"] = map[string]float64{"fintech": 1.0, "seed": 1.0}
	ds.Features["s2"] = map[string]float64{"AI": 1.0, "series-a": 1.0}
	ds.Features["u1"] = map[string]float64{"fintech": 1.0, "seed": 1.0}
	ds.Features["u2"] = map[string]float64{"fintech": 0.5, "AI": 0.5, "seed": 1.0}
	ds.Features["u3"] = map[string]float64{"AI": 1.0, "series-a": 1.0}
}

// CosineSimilarity computes similarity between two vectors
func CosineSimilarity(vec1, vec2 map[string]float64) float64 {
	dotProduct, norm1, norm2 := 0.0, 0.0, 0.0
	for k, v1 := range vec1 {
		v2, exists := vec2[k]
		if exists {
			dotProduct += v1 * v2
		}
		norm1 += v1 * v1
	}
	for _, v2 := range vec2 {
		norm2 += v2 * v2
	}
	if norm1 == 0 || norm2 == 0 {
		return 0.0
	}
	return dotProduct / (math.Sqrt(norm1) * math.Sqrt(norm2))
}

// UserBasedCollaborativeFiltering recommends based on similar users
func (ds *DataStore) UserBasedCollaborativeFiltering(userID string, k, n int) []Recommendation {
	// Compute similarities
	type sim struct {
		userID string
		score  float64
	}
	similarities := []sim{}
	for otherID := range ds.Users {
		if otherID != userID {
			simScore := CosineSimilarity(ds.Interactions[userID], ds.Interactions[otherID])
			similarities = append(similarities, sim{userID: otherID, score: simScore})
		}
	}
	// Sort and select top-k neighbors
	sort.Slice(similarities, func(i, j int) bool { return similarities[i].score > similarities[j].score })
	if len(similarities) > k {
		similarities = similarities[:k]
	}

	// Aggregate scores
	scores := make(map[string]float64)
	for _, sim := range similarities {
		for targetID, score := range ds.Interactions[sim.userID] {
			if _, exists := ds.Interactions[userID][targetID]; !exists {
				scores[targetID] += sim.score * score
			}
		}
	}

	// Convert to recommendations
	var recs []Recommendation
	for targetID := range scores {
		itemType := "startup"
		if _, exists := ds.Users[targetID]; exists {
			itemType = "user"
		}
		recs = append(recs, Recommendation{TargetID: targetID, Type: itemType, Score: scores[targetID]})
	}
	sort.Slice(recs, func(i, j int) bool { return recs[i].Score > recs[j].Score })
	if len(recs) > n {
		recs = recs[:n]
	}
	return recs
}

// ContentBasedFiltering recommends based on profile similarity
func (ds *DataStore) ContentBasedFiltering(userID string, n int) []Recommendation {
	// Build user profile
	userProfile := make(map[string]float64)
	totalWeight := 0.0
	for targetID, score := range ds.Interactions[userID] {
		for feature, value := range ds.Features[targetID] {
			userProfile[feature] += score * value
		}
		totalWeight += score
	}
	if totalWeight > 0 {
		for feature := range userProfile {
			userProfile[feature] /= totalWeight
		}
	}

	// Compute similarities
	scores := make(map[string]float64)
	for itemID := range ds.Features {
		if _, exists := ds.Interactions[userID][itemID]; !exists {
			scores[itemID] = CosineSimilarity(userProfile, ds.Features[itemID])
		}
	}

	// Convert to recommendations
	var recs []Recommendation
	for itemID := range scores {
		itemType := "startup"
		if _, exists := ds.Users[itemID]; exists {
			itemType = "user"
		}
		recs = append(recs, Recommendation{TargetID: itemID, Type: itemType, Score: scores[itemID]})
	}
	sort.Slice(recs, func(i, j int) bool { return recs[i].Score > recs[j].Score })
	if len(recs) > n {
		recs = recs[:n]
	}
	return recs
}

// HybridRecommendation combines collaborative and content-based
func (ds *DataStore) HybridRecommendation(userID string, k, n int, alpha float64) []Recommendation {
	collabRecs := ds.UserBasedCollaborativeFiltering(userID, k, n*2) // Get more to merge
	contentRecs := ds.ContentBasedFiltering(userID, n*2)

	// Combine scores
	scores := make(map[string]float64)
	for _, rec := range collabRecs {
		scores[rec.TargetID] += alpha * rec.Score
	}
	for _, rec := range contentRecs {
		scores[rec.TargetID] += (1 - alpha) * rec.Score
	}

	// Fallback to popular items for cold start
	if len(scores) < n {
		for _, itemID := range ds.GetPopularItems(n - len(scores)) {
			scores[itemID] = 0.5 // Default score
		}
	}

	// Convert to recommendations
	var recs []Recommendation
	for itemID, score := range scores {
		itemType := "startup"
		if _, exists := ds.Users[itemID]; exists {
			itemType = "user"
		}
		recs = append(recs, Recommendation{TargetID: itemID, Type: itemType, Score: score})
	}
	sort.Slice(recs, func(i, j int) bool { return recs[i].Score > recs[j].Score })
	if len(recs) > n {
		recs = recs[:n]
	}
	return recs
}

// GetPopularItems returns the most interacted items
func (ds *DataStore) GetPopularItems(n int) []string {
	scores := make(map[string]float64)
	for _, userInteractions := range ds.Interactions {
		for targetID, score := range userInteractions {
			scores[targetID] += score
		}
	}
	var items []string
	for itemID := range scores {
		items = append(items, itemID)
	}
	sort.Slice(items, func(i, j int) bool { return scores[items[i]] > scores[items[j]] })
	if len(items) > n {
		items = items[:n]
	}
	return items
}

// Concurrent recommendation processing
func (ds *DataStore) ProcessRecommendations(users []string, k, n int, alpha float64) map[string][]Recommendation {
	results := make(map[string][]Recommendation)
	resultChan := make(chan struct {
		userID string
		recs   []Recommendation
	}, len(users))

	// Spawn goroutines
	for _, userID := range users {
		go func(id string) {
			recs := ds.HybridRecommendation(id, k, n, alpha)
			resultChan <- struct {
				userID string
				recs   []Recommendation
			}{userID: id, recs: recs}
		}(userID)
	}

	// Collect results
	for i := 0; i < len(users); i++ {
		result := <-resultChan
		results[result.userID] = result.recs
	}
	return results
}

func main() {
	// Initialize and load data
	ds := NewDataStore()
	ds.LoadSampleData()

	// Test recommendation for a user
	userID := "u1" // Fintech founder
	recs := ds.HybridRecommendation(userID, 2, 3, 0.6)
	fmt.Printf("Recommendations for %s:\n", userID)
	for _, rec := range recs {
		fmt.Printf("  %s (%s): %.2f\n", rec.TargetID, rec.Type, rec.Score)
	}

	// Test concurrent recommendations
	users := []string{"u1", "u2", "u3"}
	results := ds.ProcessRecommendations(users, 2, 3, 0.6)
	fmt.Println("\nConcurrent Recommendations:")
	for userID, recs := range results {
		fmt.Printf("%s:\n", userID)
		for _, rec := range recs {
			fmt.Printf("  %s (%s): %.2f\n", rec.TargetID, rec.Type, rec.Score)
		}
	}
}

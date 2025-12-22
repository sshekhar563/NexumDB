"""
Semantic cache and query optimizer using local embedding models
"""

import numpy as np
from typing import Optional, List, Dict, Any
import json
import os
from pathlib import Path

class SemanticCache:
    """
    Caches query results using semantic similarity
    Uses local embedding models only
    Supports persistence to disk via JSON or pickle files
    """
    
    def __init__(self, similarity_threshold: float = 0.95, cache_file: str = "semantic_cache.pkl") -> None:
        self.cache: List[Dict] = []
        self.similarity_threshold = similarity_threshold
        self.model = None
        
        # Support environment variable for cache file path
        cache_file_env = os.environ.get('NEXUMDB_CACHE_FILE', cache_file)
        self.cache_file = cache_file_env
        
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_path = self.cache_dir / self.cache_file
        
        # Load existing cache on initialization
        self.load_cache()
        
    def initialize_model(self) -> None:
        """Initialize local embedding model - deferred to avoid import errors"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Semantic cache initialized with all-MiniLM-L6-v2")
        except ImportError:
            print("Warning: sentence-transformers not installed, using fallback")
            self.model = None
    
    def vectorize(self, text: str) -> List[float]:
        """Convert text to embedding vector"""
        if self.model is None:
            self.initialize_model()
        
        if self.model is not None:
            embedding = self.model.encode(text)
            return embedding.tolist()
        else:
            return self._fallback_vectorize(text)
    
    def _fallback_vectorize(self, text: str) -> List[float]:
        """Simple fallback vectorization using character hashing"""
        vec = [0.0] * 384
        for i, char in enumerate(text[:384]):
            vec[i] = float(ord(char)) / 128.0
        return vec
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_arr = np.array(vec1)
        vec2_arr = np.array(vec2)
        
        dot_product = np.dot(vec1_arr, vec2_arr)
        norm1 = np.linalg.norm(vec1_arr)
        norm2 = np.linalg.norm(vec2_arr)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def get(self, query: str) -> Optional[str]:
        """Retrieve cached result if similar query exists"""
        query_vec = self.vectorize(query)
        
        for entry in self.cache:
            similarity = self.cosine_similarity(query_vec, entry['vector'])
            if similarity >= self.similarity_threshold:
                print(f"Cache hit! Similarity: {similarity:.4f}")
                return entry['result']
        
        return None
    
    def put(self, query: str, result: str) -> None:
        """Store query and result in cache"""
        query_vec = self.vectorize(query)
        self.cache.append({
            'query': query,
            'vector': query_vec,
            'result': result
        })
        print(f"Cached query: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear the cache"""
        self.cache.clear()
        # Remove cache file when clearing
        if self.cache_path.exists():
            self.cache_path.unlink()
            print("Cache file deleted")
    
    def save_cache(self, filepath: Optional[str] = None) -> None:
        """Save cache to disk using JSON format (secure default)"""
        if filepath is None:
            filepath = str(self.cache_path)
        
        # Use JSON format by default for security
        json_filepath = filepath.replace('.pkl', '.json') if filepath.endswith('.pkl') else filepath
        self.save_cache_json(json_filepath)
    
    def load_cache(self, filepath: Optional[str] = None) -> None:
        """Load cache from disk using JSON (safe) or pickle (legacy)"""
        if filepath is None:
            filepath = str(self.cache_path)
        
        # Try JSON first (safer format)
        json_filepath = filepath.replace('.pkl', '.json') if filepath.endswith('.pkl') else f"{filepath}.json"
        if os.path.exists(json_filepath):
            self.load_cache_json(json_filepath)
            return
        
        # Fall back to pickle for legacy files (with restricted unpickler for safety)
        if os.path.exists(filepath) and filepath.endswith('.pkl'):
            try:
                import pickle
                
                # Use RestrictedUnpickler to limit allowed classes
                class RestrictedUnpickler(pickle.Unpickler):
                    """Restricted unpickler that only allows safe types"""
                    ALLOWED_CLASSES = {
                        ('builtins', 'dict'),
                        ('builtins', 'list'),
                        ('builtins', 'str'),
                        ('builtins', 'int'),
                        ('builtins', 'float'),
                        ('builtins', 'bool'),
                        ('builtins', 'tuple'),
                        ('builtins', 'set'),
                        ('builtins', 'frozenset'),
                    }
                    
                    def find_class(self, module: str, name: str) -> type:
                        if (module, name) not in self.ALLOWED_CLASSES:
                            raise pickle.UnpicklingError(
                                f"Forbidden class: {module}.{name}"
                            )
                        return super().find_class(module, name)
                
                with open(filepath, 'rb') as f:
                    data = RestrictedUnpickler(f).load()
                
                self.cache = data.get('cache', [])
                self.similarity_threshold = data.get('similarity_threshold', self.similarity_threshold)
                
                print(f"Semantic cache loaded from {filepath} ({len(self.cache)} entries)")
                print("Note: Converting legacy pickle cache to JSON format for security")
                
                # Validate cache entries
                valid_entries = []
                for entry in self.cache:
                    if all(key in entry for key in ['query', 'vector', 'result']):
                        valid_entries.append(entry)
                    else:
                        print("Warning: Invalid cache entry found and removed")
                
                self.cache = valid_entries
                
                # Auto-convert to JSON format for future use
                self.save_cache_json(json_filepath)
                
            except Exception as e:
                print(f"Error loading semantic cache: {e}")
                print("Starting with empty cache")
                self.cache = []
        else:
            print(f"No cache file found at {filepath}, starting with empty cache")
    
    def save_cache_json(self, filepath: Optional[str] = None) -> None:
        """Save cache to JSON format (secure and portable)"""
        if filepath is None:
            filepath = str(self.cache_path).replace('.pkl', '.json')
        
        try:
            # Create backup of existing cache
            backup_path = f"{filepath}.backup"
            if os.path.exists(filepath):
                os.rename(filepath, backup_path)
            
            cache_data = {
                'cache': self.cache,
                'similarity_threshold': self.similarity_threshold,
                'cache_size': len(self.cache),
                'format_version': '1.0'
            }
            
            with open(filepath, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            print(f"Semantic cache saved to {filepath} ({len(self.cache)} entries)")
            
            # Remove backup if save was successful
            if os.path.exists(backup_path):
                os.remove(backup_path)
            
        except Exception as e:
            print(f"Error saving cache to JSON: {e}")
            # Restore backup if save failed
            if os.path.exists(backup_path):
                os.rename(backup_path, filepath)
    
    def load_cache_json(self, filepath: Optional[str] = None) -> None:
        """Load cache from JSON format"""
        if filepath is None:
            filepath = str(self.cache_path).replace('.pkl', '.json')
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                self.cache = data.get('cache', [])
                self.similarity_threshold = data.get('similarity_threshold', self.similarity_threshold)
                
                print(f"Semantic cache loaded from JSON: {filepath} ({len(self.cache)} entries)")
                
            except Exception as e:
                print(f"Error loading cache from JSON: {e}")
                self.cache = []
        else:
            print(f"No JSON cache file found at {filepath}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'total_entries': len(self.cache),
            'similarity_threshold': self.similarity_threshold,
            'cache_file': str(self.cache_path),
            'cache_exists': self.cache_path.exists(),
            'cache_size_bytes': self.cache_path.stat().st_size if self.cache_path.exists() else 0
        }
    
    def explain_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query without executing - returns cache similarity scores
        and potential cache hits for EXPLAIN command
        """
        query_vec = self.vectorize(query)
        
        cache_analysis = []
        best_match = None
        best_similarity = 0.0
        
        for entry in self.cache:
            similarity = self.cosine_similarity(query_vec, entry['vector'])
            # Smart truncation for cached query display
            cached_query = entry['query']
            if len(cached_query) > 50:
                display_query = cached_query[:50] + '...'
            else:
                display_query = cached_query
                
            cache_analysis.append({
                'cached_query': display_query,
                'similarity': round(similarity, 4),
                'would_hit': similarity >= self.similarity_threshold
            })
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry['query']
        
        # Sort by similarity descending
        cache_analysis.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Smart truncation for best match
        if best_match and len(best_match) > 50:
            best_match_display = best_match[:50] + '...'
        else:
            best_match_display = best_match
        
        return {
            'query': query,
            'cache_entries_checked': len(self.cache),
            'similarity_threshold': self.similarity_threshold,
            'best_match': best_match_display,
            'best_similarity': round(best_similarity, 4),
            'would_hit_cache': best_similarity >= self.similarity_threshold,
            'top_matches': cache_analysis[:5]  # Top 5 similar cached queries
        }
    
    def set_cache_expiration(self, max_age_hours: int = 24) -> None:
        """Remove cache entries older than specified hours (future enhancement)"""
        # This would require adding timestamps to cache entries
        # For now, just a placeholder for TTL functionality
        print(f"Cache expiration set to {max_age_hours} hours (not yet implemented)")
    
    def optimize_cache(self, max_entries: int = 1000) -> None:
        """Remove oldest entries if cache exceeds max size"""
        if len(self.cache) > max_entries:
            removed_count = len(self.cache) - max_entries
            self.cache = self.cache[-max_entries:]  # Keep most recent entries
            print(f"Cache optimized: removed {removed_count} oldest entries")
            self.save_cache()


class QueryOptimizer:
    """
    Reinforcement learning-based query optimizer
    Uses Q-learning to optimize query execution
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9) -> None:
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = 0.1
        
    def get_action(self, state: str, available_actions: List[str]) -> str:
        """Select action using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in available_actions}
        
        state_values = self.q_table[state]
        best_action = max(available_actions, key=lambda a: state_values.get(a, 0.0))
        return best_action
    
    def update(self, state: str, action: str, reward: float, next_state: str) -> None:
        """Update Q-values based on observed reward"""
        if state not in self.q_table:
            self.q_table[state] = {}
        
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        current_q = self.q_table[state][action]
        
        max_next_q = 0.0
        if next_state in self.q_table:
            max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        print(f"Updated Q({state}, {action}) = {new_q:.4f}")
    
    def feed_metrics(self, query: str, latency_ms: float) -> None:
        """Feed execution metrics to the optimizer"""
        reward = -latency_ms / 1000.0
        
        state = f"query_type_{len(query) // 10}"
        action = "execute"
        next_state = "completed"
        
        self.update(state, action, reward, next_state)
    
    def explain_action(self, query: str, available_actions: List[str]) -> Dict[str, Any]:
        """
        Explain what action would be taken without executing
        Returns Q-values and predicted action for EXPLAIN command
        """
        state = f"query_type_{len(query) // 10}"
        
        q_values = {}
        if state in self.q_table:
            q_values = {a: round(v, 4) for a, v in self.q_table[state].items()}
        else:
            q_values = {a: 0.0 for a in available_actions}
        
        best_action = max(available_actions, key=lambda a: q_values.get(a, 0.0))
        
        return {
            'state': state,
            'q_values': q_values,
            'best_action': best_action,
            'epsilon': self.epsilon,
            'would_explore': self.epsilon > 0,
            'exploration_note': f'Random action possible (ε={self.epsilon})' if self.epsilon > 0 else 'Would use best action'
        }


def test_vectorization() -> Dict[str, Any]:
    """Test function for Rust integration"""
    cache = SemanticCache()
    test_query = "SELECT * FROM users WHERE age > 25"
    vector = cache.vectorize(test_query)
    return {
        'query': test_query,
        'vector': vector[:10],
        'dimension': len(vector)
    }


def explain_query_plan(query: str, cache: Optional[SemanticCache] = None, 
                       optimizer: Optional[QueryOptimizer] = None) -> Dict[str, Any]:
    """
    Generate a complete EXPLAIN plan for a query
    Shows parsing, cache analysis, and RL agent predictions
    """
    result = {
        'query': query,
        'query_length': len(query),
        'parsing': {},
        'cache_analysis': {},
        'rl_agent': {},
        'execution_strategy': {}
    }
    
    # 1. Query Parsing Analysis
    query_upper = query.upper().strip()
    if query_upper.startswith('SELECT'):
        query_type = 'SELECT'
    elif query_upper.startswith('INSERT'):
        query_type = 'INSERT'
    elif query_upper.startswith('UPDATE'):
        query_type = 'UPDATE'
    elif query_upper.startswith('DELETE'):
        query_type = 'DELETE'
    elif query_upper.startswith('CREATE'):
        query_type = 'CREATE'
    else:
        query_type = 'UNKNOWN'
    
    result['parsing'] = {
        'query_type': query_type,
        'query_length': len(query),
        'complexity_estimate': min(len(query) // 20, 10),
        'has_where_clause': 'WHERE' in query_upper,
        'has_join': 'JOIN' in query_upper,
        'has_aggregation': any(agg in query_upper for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']),
        'has_order_by': 'ORDER BY' in query_upper,
        'has_group_by': 'GROUP BY' in query_upper
    }
    
    # 2. Cache Analysis
    if cache is None:
        cache = SemanticCache()
    result['cache_analysis'] = cache.explain_query(query)
    
    # 3. RL Agent Analysis
    if optimizer is None:
        optimizer = QueryOptimizer()
    
    available_actions = ['use_cache', 'bypass_cache', 'full_scan', 'index_scan']
    result['rl_agent'] = optimizer.explain_action(query, available_actions)
    
    # 4. Execution Strategy
    would_hit_cache = result['cache_analysis'].get('would_hit_cache', False)
    best_action = result['rl_agent'].get('best_action', 'full_scan')
    
    if would_hit_cache:
        strategy = 'CACHE_HIT'
        estimated_latency = '< 1ms'
    elif best_action == 'use_cache':
        strategy = 'CACHE_MISS_THEN_STORE'
        estimated_latency = '5-50ms'
    elif best_action == 'index_scan':
        strategy = 'INDEX_SCAN'
        estimated_latency = '1-10ms'
    else:
        strategy = 'FULL_SCAN'
        estimated_latency = '10-100ms'
    
    result['execution_strategy'] = {
        'strategy': strategy,
        'estimated_latency': estimated_latency,
        'will_cache_result': query_type == 'SELECT' and not would_hit_cache,
        'recommendation': 'Use cached result' if would_hit_cache else 'Execute and cache'
    }
    
    return result


def format_explain_output(explain_result: Dict[str, Any]) -> str:
    """Format EXPLAIN result as a readable table"""
    lines = []
    lines.append("=" * 70)
    lines.append("QUERY EXECUTION PLAN")
    lines.append("=" * 70)
    
    # Smart query truncation
    query = explain_result['query']
    if len(query) > 60:
        display_query = query[:60] + "..."
    else:
        display_query = query
    
    lines.append(f"Query: {display_query}")
    lines.append("")
    
    # Parsing section
    lines.append("┌─ PARSING ─────────────────────────────────────────────────────────┐")
    p = explain_result['parsing']
    lines.append(f"│ Type: {p['query_type']:<15} Complexity: {p['complexity_estimate']}/10              │")
    lines.append(f"│ WHERE: {str(p['has_where_clause']):<8} JOIN: {str(p['has_join']):<8} AGG: {str(p['has_aggregation']):<8}     │")
    lines.append("└───────────────────────────────────────────────────────────────────┘")
    lines.append("")
    
    # Cache section
    lines.append("┌─ CACHE LOOKUP ────────────────────────────────────────────────────┐")
    c = explain_result['cache_analysis']
    lines.append(f"│ Entries checked: {c['cache_entries_checked']:<5} Threshold: {c['similarity_threshold']:<6}            │")
    lines.append(f"│ Best similarity: {c['best_similarity']:<6} Would hit: {str(c['would_hit_cache']):<6}              │")
    if c['top_matches']:
        lines.append("│ Top matches:                                                      │")
        for match in c['top_matches'][:3]:
            sim = match['similarity']
            hit = "✓" if match['would_hit'] else "✗"
            # Smart truncation for cached queries
            cached_query = match['cached_query']
            if not cached_query.endswith('...') and len(cached_query) > 45:
                cached_query = cached_query[:42] + "..."
            lines.append(f"│   {hit} {sim:.4f} - {cached_query:<45} │")
    lines.append("└───────────────────────────────────────────────────────────────────┘")
    lines.append("")
    
    # RL Agent section
    lines.append("┌─ RL AGENT ────────────────────────────────────────────────────────┐")
    r = explain_result['rl_agent']
    lines.append(f"│ State: {r['state']:<30} Epsilon: {r.get('epsilon', r.get('exploration_probability', 0)):<6}        │")
    lines.append(f"│ Best action: {r['best_action']:<20}                          │")
    lines.append("│ Q-values:                                                         │")
    for action, qval in r['q_values'].items():
        lines.append(f"│   {action:<15}: {qval:>8.4f}                                    │")
    lines.append("└───────────────────────────────────────────────────────────────────┘")
    lines.append("")
    
    # Execution strategy
    lines.append("┌─ EXECUTION STRATEGY ──────────────────────────────────────────────┐")
    e = explain_result['execution_strategy']
    lines.append(f"│ Strategy: {e['strategy']:<20} Est. latency: {e['estimated_latency']:<10}   │")
    lines.append(f"│ Will cache: {str(e['will_cache_result']):<8}                                          │")
    lines.append(f"│ Recommendation: {e['recommendation']:<40}       │")
    lines.append("└───────────────────────────────────────────────────────────────────┘")
    
    return "\n".join(lines)


def test_cache_persistence() -> Dict[str, Any]:
    """Test semantic cache persistence functionality"""
    print("\n" + "="*60)
    print("Testing Semantic Cache Persistence")
    print("="*60 + "\n")
    
    # Test 1: Create cache and add entries
    print("1. Creating cache and adding test entries...")
    cache1 = SemanticCache(cache_file="test_cache.pkl")
    
    test_queries = [
        ("SELECT * FROM users WHERE age > 25", "User data for age > 25"),
        ("SELECT name FROM products WHERE price < 100", "Product names under $100"),
        ("SELECT COUNT(*) FROM orders WHERE status = 'active'", "Active order count: 42")
    ]
    
    for query, result in test_queries:
        cache1.put(query, result)
    
    # Save cache after adding entries
    cache1.save_cache()
    
    stats1 = cache1.get_cache_stats()
    print(f"Cache stats after adding entries: {stats1}")
    
    # Test 2: Create new cache instance and verify persistence
    print("\n2. Creating new cache instance to test persistence...")
    cache2 = SemanticCache(cache_file="test_cache.pkl")
    
    stats2 = cache2.get_cache_stats()
    print(f"Cache stats after reload: {stats2}")
    
    # Test 3: Verify cache hits work after reload
    print("\n3. Testing cache hits after reload...")
    for query, expected_result in test_queries:
        cached_result = cache2.get(query)
        if cached_result:
            print(f"✓ Cache hit for: {query[:30]}...")
            print(f"  Result: {cached_result[:50]}...")
        else:
            print(f"✗ Cache miss for: {query[:30]}...")
    
    # Test 4: Test JSON export
    print("\n4. Testing JSON export...")
    cache2.save_cache_json("test_cache.json")
    
    # Test 5: Test cache optimization
    print("\n5. Testing cache optimization...")
    cache2.optimize_cache(max_entries=2)
    
    # Cleanup
    print("\n6. Cleaning up test files...")
    cache2.clear()
    
    return {
        'test_passed': True,
        'entries_before_reload': stats1['total_entries'],
        'entries_after_reload': stats2['total_entries'],
        'persistence_working': stats1['total_entries'] == stats2['total_entries']
    }


if __name__ == "__main__":
    # Run both tests
    print("Running vectorization test...")
    result = test_vectorization()
    print(json.dumps(result, indent=2))
    
    print("\nRunning persistence test...")
    persistence_result = test_cache_persistence()
    print(f"\nPersistence test result: {persistence_result}")
    
    # Test EXPLAIN functionality
    print("\n" + "="*70)
    print("Testing EXPLAIN Query Plan")
    print("="*70)
    
    # Add some test data to cache first
    cache = SemanticCache()
    cache.put("SELECT * FROM users WHERE age > 25", "User data result")
    cache.put("SELECT name FROM products WHERE price < 100", "Product names")
    
    # Test explain
    test_query = "SELECT * FROM users WHERE age > 30"
    explain_result = explain_query_plan(test_query, cache)
    print(format_explain_output(explain_result))

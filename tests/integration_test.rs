use nexum_core::{StorageEngine, Parser, Executor};

#[test]
fn test_where_clause_filtering() {
    let storage = StorageEngine::memory().unwrap();
    let executor = Executor::new(storage);
    
    let create = Parser::parse("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)").unwrap();
    executor.execute(create).unwrap();
    
    let insert = Parser::parse(
        "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30), (2, 'Bob', 22), (3, 'Charlie', 28)"
    ).unwrap();
    executor.execute(insert).unwrap();
    
    let select = Parser::parse("SELECT * FROM users WHERE age > 25").unwrap();
    let result = executor.execute(select).unwrap();
    
    match result {
        nexum_core::executor::ExecutionResult::Selected { rows, .. } => {
            println!("Filtered rows: {:?}", rows);
            assert_eq!(rows.len(), 2);
        }
        _ => panic!("Expected Selected result"),
    }
}

#[test]
fn test_where_clause_with_and() {
    let storage = StorageEngine::memory().unwrap();
    let executor = Executor::new(storage);
    
    let create = Parser::parse("CREATE TABLE products (id INTEGER, name TEXT, price INTEGER)").unwrap();
    executor.execute(create).unwrap();
    
    let insert = Parser::parse(
        "INSERT INTO products (id, name, price) VALUES (1, 'Laptop', 1000), (2, 'Mouse', 25), (3, 'Keyboard', 75)"
    ).unwrap();
    executor.execute(insert).unwrap();
    
    let select = Parser::parse("SELECT * FROM products WHERE price > 50 AND price < 500").unwrap();
    let result = executor.execute(select).unwrap();
    
    match result {
        nexum_core::executor::ExecutionResult::Selected { rows, .. } => {
            println!("Filtered rows: {:?}", rows);
            assert_eq!(rows.len(), 1);
        }
        _ => panic!("Expected Selected result"),
    }
}

#[test]
fn test_where_clause_text_equality() {
    let storage = StorageEngine::memory().unwrap();
    let executor = Executor::new(storage);
    
    let create = Parser::parse("CREATE TABLE users (id INTEGER, name TEXT)").unwrap();
    executor.execute(create).unwrap();
    
    let insert = Parser::parse(
        "INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Alice')"
    ).unwrap();
    executor.execute(insert).unwrap();
    
    let select = Parser::parse("SELECT * FROM users WHERE name = 'Alice'").unwrap();
    let result = executor.execute(select).unwrap();
    
    match result {
        nexum_core::executor::ExecutionResult::Selected { rows, .. } => {
            println!("Filtered rows: {:?}", rows);
            assert_eq!(rows.len(), 2);
        }
        _ => panic!("Expected Selected result"),
    }
}

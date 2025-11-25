use crate::sql::types::Value;
use sqlparser::ast::{BinaryOperator, Expr, Value as SqlValue};
use anyhow::{anyhow, Result};

pub struct ExpressionEvaluator {
    column_names: Vec<String>,
}

impl ExpressionEvaluator {
    pub fn new(column_names: Vec<String>) -> Self {
        Self { column_names }
    }

    pub fn evaluate(&self, expr: &Expr, row_values: &[Value]) -> Result<bool> {
        match expr {
            Expr::BinaryOp { left, op, right } => {
                self.evaluate_binary_op(left, op, right, row_values)
            }
            Expr::Identifier(ident) => {
                let col_name = ident.value.as_str();
                let idx = self.column_names.iter()
                    .position(|name| name == col_name)
                    .ok_or_else(|| anyhow!("Column {} not found", col_name))?;
                
                match &row_values[idx] {
                    Value::Boolean(b) => Ok(*b),
                    _ => Err(anyhow!("Expected boolean value for identifier")),
                }
            }
            _ => Err(anyhow!("Unsupported expression type: {:?}", expr)),
        }
    }

    fn evaluate_binary_op(
        &self,
        left: &Expr,
        op: &BinaryOperator,
        right: &Expr,
        row_values: &[Value],
    ) -> Result<bool> {
        match op {
            BinaryOperator::And => {
                let left_result = self.evaluate(left, row_values)?;
                let right_result = self.evaluate(right, row_values)?;
                Ok(left_result && right_result)
            }
            BinaryOperator::Or => {
                let left_result = self.evaluate(left, row_values)?;
                let right_result = self.evaluate(right, row_values)?;
                Ok(left_result || right_result)
            }
            BinaryOperator::Gt
            | BinaryOperator::Lt
            | BinaryOperator::GtEq
            | BinaryOperator::LtEq
            | BinaryOperator::Eq
            | BinaryOperator::NotEq => {
                let left_val = self.extract_value(left, row_values)?;
                let right_val = self.extract_value(right, row_values)?;
                self.compare_values(&left_val, op, &right_val)
            }
            _ => Err(anyhow!("Unsupported operator: {:?}", op)),
        }
    }

    fn extract_value(&self, expr: &Expr, row_values: &[Value]) -> Result<Value> {
        match expr {
            Expr::Identifier(ident) => {
                let col_name = ident.value.as_str();
                let idx = self.column_names.iter()
                    .position(|name| name == col_name)
                    .ok_or_else(|| anyhow!("Column {} not found", col_name))?;
                Ok(row_values[idx].clone())
            }
            Expr::Value(sql_val) => self.convert_sql_value(sql_val),
            _ => Err(anyhow!("Cannot extract value from expression: {:?}", expr)),
        }
    }

    fn convert_sql_value(&self, sql_val: &SqlValue) -> Result<Value> {
        match sql_val {
            SqlValue::Number(n, _) => {
                if n.contains('.') {
                    Ok(Value::Float(n.parse()?))
                } else {
                    Ok(Value::Integer(n.parse()?))
                }
            }
            SqlValue::SingleQuotedString(s) | SqlValue::DoubleQuotedString(s) => {
                Ok(Value::Text(s.clone()))
            }
            SqlValue::Boolean(b) => Ok(Value::Boolean(*b)),
            SqlValue::Null => Ok(Value::Null),
            _ => Err(anyhow!("Unsupported SQL value: {:?}", sql_val)),
        }
    }

    fn compare_values(&self, left: &Value, op: &BinaryOperator, right: &Value) -> Result<bool> {
        match (left, right) {
            (Value::Integer(l), Value::Integer(r)) => {
                Ok(match op {
                    BinaryOperator::Eq => l == r,
                    BinaryOperator::NotEq => l != r,
                    BinaryOperator::Gt => l > r,
                    BinaryOperator::Lt => l < r,
                    BinaryOperator::GtEq => l >= r,
                    BinaryOperator::LtEq => l <= r,
                    _ => return Err(anyhow!("Invalid operator for integers")),
                })
            }
            (Value::Float(l), Value::Float(r)) => {
                Ok(match op {
                    BinaryOperator::Eq => (l - r).abs() < f64::EPSILON,
                    BinaryOperator::NotEq => (l - r).abs() >= f64::EPSILON,
                    BinaryOperator::Gt => l > r,
                    BinaryOperator::Lt => l < r,
                    BinaryOperator::GtEq => l >= r,
                    BinaryOperator::LtEq => l <= r,
                    _ => return Err(anyhow!("Invalid operator for floats")),
                })
            }
            (Value::Text(l), Value::Text(r)) => {
                Ok(match op {
                    BinaryOperator::Eq => l == r,
                    BinaryOperator::NotEq => l != r,
                    BinaryOperator::Gt => l > r,
                    BinaryOperator::Lt => l < r,
                    BinaryOperator::GtEq => l >= r,
                    BinaryOperator::LtEq => l <= r,
                    _ => return Err(anyhow!("Invalid operator for text")),
                })
            }
            (Value::Boolean(l), Value::Boolean(r)) => {
                Ok(match op {
                    BinaryOperator::Eq => l == r,
                    BinaryOperator::NotEq => l != r,
                    _ => return Err(anyhow!("Invalid operator for booleans")),
                })
            }
            (Value::Null, Value::Null) => {
                Ok(match op {
                    BinaryOperator::Eq => true,
                    BinaryOperator::NotEq => false,
                    _ => return Err(anyhow!("Invalid operator for nulls")),
                })
            }
            _ => Err(anyhow!("Type mismatch in comparison: {:?} vs {:?}", left, right)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser;

    #[test]
    fn test_simple_comparison() {
        let column_names = vec!["id".to_string(), "name".to_string(), "age".to_string()];
        let evaluator = ExpressionEvaluator::new(column_names);
        
        let row_values = vec![
            Value::Integer(1),
            Value::Text("Alice".to_string()),
            Value::Integer(30),
        ];
        
        let sql = "age > 25";
        let dialect = GenericDialect {};
        let ast = Parser::parse_sql(&dialect, &format!("SELECT * FROM t WHERE {}", sql)).unwrap();
        
        if let sqlparser::ast::Statement::Query(query) = &ast[0] {
            if let sqlparser::ast::SetExpr::Select(select) = &*query.body {
                if let Some(where_expr) = &select.selection {
                    let result = evaluator.evaluate(where_expr, &row_values).unwrap();
                    assert!(result);
                }
            }
        }
    }

    #[test]
    fn test_and_operator() {
        let column_names = vec!["id".to_string(), "age".to_string()];
        let evaluator = ExpressionEvaluator::new(column_names);
        
        let row_values = vec![Value::Integer(1), Value::Integer(30)];
        
        let sql = "id = 1 AND age > 25";
        let dialect = GenericDialect {};
        let ast = Parser::parse_sql(&dialect, &format!("SELECT * FROM t WHERE {}", sql)).unwrap();
        
        if let sqlparser::ast::Statement::Query(query) = &ast[0] {
            if let sqlparser::ast::SetExpr::Select(select) = &*query.body {
                if let Some(where_expr) = &select.selection {
                    let result = evaluator.evaluate(where_expr, &row_values).unwrap();
                    assert!(result);
                }
            }
        }
    }

    #[test]
    fn test_text_comparison() {
        let column_names = vec!["name".to_string()];
        let evaluator = ExpressionEvaluator::new(column_names);
        
        let row_values = vec![Value::Text("Alice".to_string())];
        
        let sql = "name = 'Alice'";
        let dialect = GenericDialect {};
        let ast = Parser::parse_sql(&dialect, &format!("SELECT * FROM t WHERE {}", sql)).unwrap();
        
        if let sqlparser::ast::Statement::Query(query) = &ast[0] {
            if let sqlparser::ast::SetExpr::Select(select) = &*query.body {
                if let Some(where_expr) = &select.selection {
                    let result = evaluator.evaluate(where_expr, &row_values).unwrap();
                    assert!(result);
                }
            }
        }
    }
}

use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum MetricValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MetricType {
    Bool,
    Int,
    Float,
    Str,
}

impl From<&MetricValue> for MetricType {
    fn from(v: &MetricValue) -> Self {
        match v {
            MetricValue::Bool(_)  => MetricType::Bool,
            MetricValue::Int(_)   => MetricType::Int,
            MetricValue::Float(_) => MetricType::Float,
            MetricValue::Str(_)   => MetricType::Str,
        }
    }
}
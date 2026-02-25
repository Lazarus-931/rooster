

pub struct Session {
    pub name: String,
    pub id: usize
}


impl Session {
    pub fn new(name: String) -> Self {
        Self { name, id: 0 }
    }
}
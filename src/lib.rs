use ndarray::{arr1, Array2};
use rusqlite::{params, Connection};

fn softmax(arr: Array2<f32>) -> Array2<f32> {
    let max: f32 = arr.iter().cloned().fold(arr[(0, 0)], f32::max);
    let logits: Array2<f32> = arr.mapv(|x| (x - max).exp());
    let sum: f32 = logits.sum();

    return logits/sum;
}

fn to_bytes(values: &[f32]) -> Vec<u8> {
    let mut r = Vec::with_capacity(values.len()*4);
    for &v in values {
        r.extend_from_slice(&v.to_le_bytes());
    }
    return r;
}

fn to_f32(bytes: &[u8]) -> Vec<f32> {
    let mut r = Vec::with_capacity(bytes.len()/4);
    for chunk in bytes.chunks_exact(4) {
        let a: [u8;4] = chunk.try_into().unwrap();
        r.push(f32::from_le_bytes(a));
    }

    return r;
}

pub struct HopfieldNet {
    x: Array2<f32>,
    beta: f32
}

pub fn hopfield_net_init(x: Array2<f32>, beta: Option<f32>) -> HopfieldNet {
    HopfieldNet {
        x: x,
        beta: beta.unwrap_or(100.0),
    }
}

impl HopfieldNet {
    pub fn update_rule(&self, eps: Array2<f32>) -> Array2<f32> {
        return softmax(self.beta * eps.dot(&self.x.t())).dot(&self.x)
    }

    pub fn converge(&self, mut eps: Array2<f32>) -> Array2<f32> {
        let mut pre = Array2::<f32>::zeros((1, eps.dim().1));

        while &pre != &eps {
            pre = eps.clone();
            eps = self.update_rule(eps)

        }
        return pre;
    }

}

pub struct VectorDatabase {
    con: Connection
}

pub fn vectordb_init(file: &str) -> VectorDatabase {
    let mut db = VectorDatabase {
        con: Connection::open(file).unwrap(),
    };
    db.setup();
    return db
}

impl VectorDatabase {
    pub fn setup(&mut self) {
        let _ = self.con.execute("CREATE TABLE IF NOT EXISTS documents(embeddings BLOB, text TEXT)", []);
    }

    pub fn add(&self, embedding: Vec<f32>, text: &str) {
        let _ = self.con.execute("INSERT INTO documents VALUES(?, ?)", params![to_bytes(&embedding), text]);
    }

    pub fn get(&self, embedding: Vec<f32>) -> String {
        let mut query = self.con.prepare("SELECT text FROM documents WHERE embeddings=(?1)").unwrap();
        let mut r = query.query([to_bytes(&embedding)]).unwrap();

        while let Some(row) = r.next().unwrap() {
            let t: String = row.get(0).unwrap();
            return t
        }

        return String::new()
    }

    pub fn get_all_embeddings(&self) -> Array2<f32> {
        let mut query = self.con.prepare("SELECT embeddings FROM documents").unwrap();
        let mut r = query.query([]).unwrap();

        let mut matrix = Array2::<f32>::zeros((0, 0));
        while let Some(row) = r.next().unwrap() {
            let bytes: Vec<u8> = row.get(0).unwrap();
            matrix.push_column(arr1(&to_f32(&bytes)).t()).unwrap();
        }

        return matrix
    }

    pub fn close(self) {
        self.con.close().unwrap();
    }
}

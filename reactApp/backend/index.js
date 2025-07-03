const express = require('express');
const cors = require('cors');
const mysql = require('mysql2');
const bcrypt = require('bcryptjs');

const app = express();
const PORT = 5002;

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended : true}));
require('dotenv').config();

const db = mysql.createConnection({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME
});

db.connect(err => {
    if (err) throw err;
    console.log('Connected to MySql Database');
});

app.post('/api/register', async (req,res)=>{
    const { username, password, email, plate_number }=req.body;

    try{
        const hashedPassword = await bcrypt.hash(password, 10);

        const sql = 'INSERT INTO Users (username, password_hash, email, plate_number) VALUES (?, ?, ?, ?)';

        db.query(sql, [username, hashedPassword, email, plate_number], (err, result)=>{
            if (err) {
                if (err.code === 'ER_DUP_ENTRY') {
                    return res.status(409).json({ message: 'Username, email or plate already exists' });
                }
                return res.status(500).json({ message: 'Database error', error: err });
            }
            res.json({ message: 'User registered successfully', userId: result.insertId });
        });
    }catch (error) {
        res.status(500).json({ message: 'Error hashing password', error });
    }
});

app.listen(PORT, () => {
    console.log(`Backend running on http://localhost:${PORT}`)
});
const express = require('express');
const cors = require('cors');
const mysql = require('mysql2');
const bcrypt = require('bcryptjs');
const session = require('express-session');
const MySQLStore = require('express-mysql-session')(session);

const app = express();
const PORT = 5002;

app.use(cors({
    origin: 'http://localhost:5173',
    credentials: true
}));
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

const sessionStore = new MySQLStore({}, db.promise());

app.use(session({
  secret: 'your_secret_here',
  resave: false,
  saveUninitialized: false,
  store: sessionStore,
  cookie: { secure: false, httpOnly: true, maxAge: 1000 * 60 * 60 } // 1h
}));

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

app.post('/api/login', (req,res)=>{
    const {username, password} = req.body;

    const sql = 'SELECT * FROM Users WHERE username = ?';
    db.query(sql, [username], async (err, results)=>{
        if (err) return res.status(500).json({ message: 'Database error', error: err });
        if (results.length === 0) return res.status(401).json({ message: 'Invalid credentials' });

        const user = results[0];
        
        const isMatch = await bcrypt.compare(password, user.password_hash);
        if (!isMatch) return res.status(401).json({ message: 'Invalid credentials' });

        req.session.userId=user.id;

        res.json({ message: 'Login successful', user: { id: user.id, username: user.username } });
    });
});

app.get('/api/profile', (req, res) => {
  if (!req.session.userId) return res.status(401).json({ message: 'Not logged in' });

  const sql = 'SELECT id, username, plate_number, email FROM Users WHERE id = ?';
  db.query(sql, [req.session.userId], (err, results) => {
    if (err || results.length === 0) return res.status(500).json({ message: 'Failed to fetch user' });
    res.json({ user: results[0] });
  });
});

app.get('/api/parking-status/:plate', async (req, res) => {
  const plate = req.params.plate;

  try {
    const [rows] = await db.promise().query(
      `SELECT entry_time, exit_time 
       FROM ParkingActivity 
       WHERE plate_number = ? 
       ORDER BY entry_time DESC 
       LIMIT 1`,
      [plate]
    );

    if (rows.length === 0) {
      return res.json({ status: 'Unknown', lastEntry: null });
    }

    const row = rows[0];
    const status = row.exit_time ? 'Outside' : 'Inside';
    res.json({ status, lastEntry: row.entry_time });

  } catch (err) {
    console.error('Error fetching parking status:', err);
    res.status(500).json({ error: 'Server error' });
  }
});

app.get('/api/amount-due/:userId', (req, res) => {
    const userId = req.params.userId;

    const sql = `
      SELECT SUM(amount) AS total_due
      FROM Payments
      WHERE user_id = ? AND paid = FALSE
    `;

    db.query(sql, [userId], (err, results) => {
        if (err) {
            console.error('SQL error:', err);
            return res.status(500).json({ error: 'Database error' });
        }

        const totalDue = results[0].total_due || 0;
        res.json({ user_id: userId, total_due: totalDue });
    });
});

app.put('/api/pay/:userId', (req, res) => {
    const userId = req.params.userId;
    const now = new Date();

    const sql = `
      UPDATE Payments
      SET paid = TRUE,
          payment_time = ?
      WHERE user_id = ? AND paid = FALSE
    `;

    db.query(sql, [now, userId], (err, result) => {
        if (err) {
            console.error('Payment update error:', err);
            return res.status(500).json({ error: 'Database error' });
        }

        res.json({ success: true, message: 'Payments marked as paid' });
    });
});

app.get('/api/recentPayments/:userId', async (req, res) => {
    const userId = req.params.userId;

    try {
      const [rows] = await db.promise().query(
        `SELECT amount, payment_time 
        FROM Payments 
        WHERE user_id = ? 
        ORDER BY payment_time DESC 
        LIMIT 2`,
        [userId]
      );

      res.json(rows);
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: 'Database error' });
  }
});

app.get('/api/visitsLastMonth/:userId', async (req,res) =>{
  const userId = req.params.userId;

  try{
    const [rows] = await db.promise().query(
      `SELECT COUNT(*) AS visit_count
       FROM ParkingActivity
       WHERE user_id = ? AND entry_time >= NOW() - INTERVAL 30 DAY
      `,
      [userId]
    );

    res.json({visits : rows[0].visit_count});
  }catch (err){
    res.status(500).json({ error: 'Database error'});
  }
});

app.listen(PORT, () => {
    console.log(`Backend running on http://localhost:${PORT}`)
});
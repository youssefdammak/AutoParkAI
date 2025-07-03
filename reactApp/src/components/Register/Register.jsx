import './Register.css'
import { useState } from 'react';
function Register(){

    const [formData, setFormData] = useState({
        username: '',
        email: '',
        password: '',
        confirmPassword: '',
        plate_number: ''
    });
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    const handleChange = e => {
        setFormData(prev => ({...prev, [e.target.name]: e.target.value}));
    };

    const handleSubmit = async (e) =>{
        e.preventDefault();

        setError('');
        setSuccess('');

        if (formData.password !== formData.confirmPassword){
            setError('password does not match');
            return;
        }

        const {username, email, password, plate_number} = formData;

        try{
            const res = await fetch('http://localhost:5002/api/register',{
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, email, password, plate_number }),
            });

            const data = await res.json();

            if (!res.ok){
                setError(data.message || 'Registration failed');
            }
            else {
                setSuccess('Registration successful! You can now log in.');
                setFormData({
                username: '',
                email: '',
                password: '',
                confirmPassword: '',
                plate_number: '',
                });
            }
        }catch (err) {
            setError('Network error: ' + err.message);
        }
    }; 
    return (
        <div className="login-bg">
        <div className="login-box">
            <h2>Register</h2>
            <form onSubmit={handleSubmit}>
            <input
                type="text"
                name="username"
                placeholder="Username"
                value={formData.username}
                onChange={handleChange}
                required
            />
            <input
                type="email"
                name="email"
                placeholder="E-Mail"
                value={formData.email}
                onChange={handleChange}
                required
            />
            <input
                type="password"
                name="password"
                placeholder="Password"
                value={formData.password}
                onChange={handleChange}
                required
            />
            <input
                type="password"
                name="confirmPassword"
                placeholder="Re-Type Password"
                value={formData.confirmPassword}
                onChange={handleChange}
                required
            />
            <input
                type="text"
                name="plate_number"
                placeholder="Car Plate Number"
                value={formData.plate_number}
                onChange={handleChange}
                required
            />
            <button type="submit">Sign Up</button>
            </form>

            {error && <p style={{ color: 'red', marginTop: '1rem' }}>{error}</p>}
            {success && <p style={{ color: 'green', marginTop: '1rem' }}>{success}</p>}

            <p className="signup-link">
            Have an account? <a href="/account/login">Login</a>
            </p>
        </div>
        </div>
  );
}

export default Register;
import './Header.css';
import { Link, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';

function Header() {
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [username, setUsername] = useState('');
    const navigate = useNavigate();

    useEffect(() => {
        fetch('http://localhost:5002/api/profile', {
            credentials: 'include',
        })
        .then(res => res.json())
        .then(data => {
            if (data.user) {
                setIsLoggedIn(true);
                setUsername(data.user.username);
            }
        })
        .catch(() => setIsLoggedIn(false));
    }, []);

    const handleLogout = async () => {
        await fetch('http://localhost:5002/api/logout', {
            method: 'POST',
            credentials: 'include',
        });
        setIsLoggedIn(false);
        navigate('/account/login');
    };

    return (
        <header>
            <div className="header-container">
                <div className="logo">
                    <i className="fas fa-parking"></i>
                    <div>
                        <h1>AutoParkAI</h1>
                        <p>Smart Parking Management</p>
                    </div>
                </div>

                <nav>
                    <ul>
                        <li><Link to="/"><i className="fas fa-home"></i> Home</Link></li>
                        <li><Link to="/map"><i className="fas fa-map-marker-alt"></i> Parking Map</Link></li>
                        
                        {isLoggedIn ? (
                            <>
                                <li><Link to="/profile"><i className="fas fa-user-circle"></i> {username}</Link></li>
                                <li><button onClick={handleLogout} style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer' }}>
                                    <i className="fas fa-sign-out-alt"></i> Logout
                                </button></li>
                            </>
                        ) : (
                            <li><Link to="/account/login"><i className="fas fa-user"></i> Account</Link></li>
                        )}
                    </ul>
                </nav>
            </div>
        </header>
    );
}

export default Header;

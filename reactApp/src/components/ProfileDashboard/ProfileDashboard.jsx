import './ProfileDashboard.css';
import { useEffect, useState } from 'react';
import Chart from 'chart.js/auto';

function ProfileDashboard() {
    const [user, setUser] = useState(null);

    useEffect(() => {
        fetch('http://localhost:5002/api/profile', {
            credentials: 'include'
        })
        .then(res => res.json())
        .then(data => {
            console.log('User:', data.user);
            setUser(data.user);
        });
    }, []);

    useEffect(() => {
        const ctx = document.getElementById('activityChart');
        if (ctx) {
        new Chart(ctx, {
            type: 'bar',
            data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [{
                label: 'Sessions',
                data: [2, 4, 3, 5, 1, 0, 2],
                backgroundColor: 'rgba(67, 97, 238, 0.6)',
                borderRadius: 5
            }],
            },
            options: {
            responsive: true,
            scales: {
                y: {
                beginAtZero: true,
                ticks: {
                    precision: 0
                }
                }
            }
            }
        });
        }
    }, []);
    if (!user) return <p>Loading profile...</p>;
    return (
        <div className="main-container">
        <section className="dashboard">
            {/* Welcome & Info Card */}
            <div className="card">
            <div className="card-header">
                <h3 className="card-title">Welcome, {user.username}</h3>
                <div className="card-icon">
                <i className="fas fa-user-circle"></i>
                </div>
            </div>
            <div className="stats">
                <div className="stat-item">
                <div className="stat-value">{user.plate_number}</div>
                <div className="stat-label">Plate Number</div>
                </div>
                <div className="stat-item">
                <div className="stat-value">10</div>
                <div className="stat-label">Visits This Month</div>
                </div>
            </div>
            </div>

            {/* Weekly Activity Chart */}
            <div className="card">
            <div className="card-header">
                <h3 className="card-title">Weekly Parking Activity</h3>
                <div className="card-icon"><i className="fas fa-chart-bar"></i></div>
            </div>
            <canvas id="activityChart" height="100"></canvas>
            </div>

            {/* Recent Payments */}
            <div className="card">
            <div className="card-header">
                <h3 className="card-title">Recent Payments</h3>
                <div className="card-icon">
                <i className="fas fa-file-invoice-dollar"></i>
                </div>
            </div>
            <div className="log-item">
                <div className="log-plate"><i className="fas fa-car"></i> ABC1234</div>
                <div className="log-time">July 2, 2025</div>
                <div className="log-status entry">$5.00</div>
            </div>
            <div className="log-item">
                <div className="log-plate"><i className="fas fa-car"></i> XYZ7890</div>
                <div className="log-time">June 30, 2025</div>
                <div className="log-status entry">$7.00</div>
            </div>
            </div>

            {/* Amount Due & Payment */}
            <div className="card">
            <div className="card-header">
                <h3 className="card-title">Amount Due</h3>
                <div className="card-icon">
                <i className="fas fa-credit-card"></i>
                </div>
            </div>
            <div className="stats" style={{ justifyContent: 'center', marginBottom: '1rem' }}>
                <div className="stat-item">
                <div className="stat-value" style={{ color: 'var(--danger)', fontSize: '2rem' }}>$12.50</div>
                <div className="stat-label">Current Balance</div>
                </div>
            </div>
            <button className="cta-button" style={{ width: '100%' }}>
                <i className="fas fa-paper-plane"></i> Pay Now
            </button>
            </div>

            {/* Parking Status */}
            <div className="card">
            <div className="card-header">
                <h3 className="card-title">Parking Status</h3>
                <div className="card-icon"><i className="fas fa-traffic-light"></i></div>
            </div>
            <div className="stats">
                <div className="stat-item">
                <div className="stat-value" style={{ color: 'green' }}>Inside</div>
                <div className="stat-label">Current Status</div>
                </div>
                <div className="stat-item">
                <div className="stat-value">10:45 AM</div>
                <div className="stat-label">Last Entry</div>
                </div>
            </div>
            </div>

            {/* Recent Sessions Table */}
            <div className="card">
            <div className="card-header">
                <h3 className="card-title">Recent Parking Sessions</h3>
                <div className="card-icon"><i className="fas fa-history"></i></div>
            </div>
            <table style={{ width: '100%', fontSize: '0.9rem' }}>
                <thead>
                <tr style={{ textAlign: 'left' }}>
                    <th>Spot</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>Duration</th>
                </tr>
                </thead>
                <tbody>
                <tr>
                    <td>A12</td>
                    <td>09:12 AM</td>
                    <td>11:45 AM</td>
                    <td>2h 33m</td>
                </tr>
                <tr>
                    <td>B03</td>
                    <td>02:00 PM</td>
                    <td>04:10 PM</td>
                    <td>2h 10m</td>
                </tr>
                </tbody>
            </table>
            </div>

            {/* Account Management */}
            <div className="card">
            <div className="card-header">
                <h3 className="card-title">Manage Account</h3>
                <div className="card-icon"><i className="fas fa-user-cog"></i></div>
            </div>
            <div className="stats" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '1rem' }}>
                <button className="cta-button outline"><i className="fas fa-edit"></i> Edit Profile</button>
                <button className="cta-button outline"><i className="fas fa-key"></i> Change Password</button>
                <button className="cta-button outline danger"><i className="fas fa-trash"></i> Delete Account</button>
            </div>
            </div>
        </section>
        </div>
    );
}

export default ProfileDashboard;

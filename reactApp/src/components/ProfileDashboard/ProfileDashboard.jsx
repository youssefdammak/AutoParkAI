import './ProfileDashboard.css';
import { useEffect, useState } from 'react';
import Chart from 'chart.js/auto';

function ProfileDashboard() {
    const [user, setUser] = useState(null);

    const [parkingStatus, setParkingStatus] = useState({ status: '', lastEntry: '' });

    const [amountDue, setAmountDue] = useState(0);

    const [recentPayments, setRecentPayments] = useState([{amount: 0, payment_time: ''},{amount: 0, payment_time: ''}]);

    const [visitsLastMonth, setVisitsLastMonth] = useState(0);

    const [weeklyActivity, setWeeklyActivity] = useState([
                                                            { day: 'Monday', sessions: 0 },
                                                            { day: 'Tuesday', sessions: 0 },
                                                            { day: 'Wednesday', sessions: 0 },
                                                            { day: 'Thursday', sessions: 0 },
                                                            { day: 'Friday', sessions: 0 },
                                                            { day: 'Saturday', sessions: 0 },
                                                            { day: 'Sunday', sessions: 0 }
                                                        ]);

    const [parkingSpot, setParkingSpot] = useState({spot : null});

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
    if (user?.plate_number) {
        fetch(`http://localhost:5002/api/parking-status/${user.plate_number}`)
        .then(res => res.json())
        .then(data => {
            setParkingStatus(data);
        });
    }
    }, [user]);

    useEffect(() => {
    if (user?.plate_number) {
        fetch(`http://localhost:5002/api/amount-due/${user.id}`)
        .then(res => res.json())
        .then(data => {
            console.log(data);
            setAmountDue(data.total_due);
        });
    }
    }, [user]);

    useEffect(() => {
        if (user?.plate_number){
            fetch(`http://localhost:5002/api/recentPayments/${user.id}`)
            .then(res => res.json())
            .then(data => {
                setRecentPayments(data);
            });
        }
    }, [user]);

    useEffect(() => {
        if (user?.plate_number){
            fetch(`http://localhost:5002/api/visitsLastMonth/${user.id}`)
            .then(res => res.json())
            .then(data => {
                setVisitsLastMonth(data.visits);
            });
        }
    }, [user]);

    useEffect(() => {
        if (user?.plate_number){
            fetch(`http://localhost:5002/api/weekly-activity/${user.id}`)
            .then(res => res.json())
            .then(data => {
                setWeeklyActivity(data);
            });
        }
    }, [user]);

    useEffect(() => {
        if (user?.plate_number) {
            fetch(`http://localhost:5002/api/parking-spot/${user.plate_number}`)
            .then(res => res.json())
            .then(data => {
                setParkingSpot(data);
            });
        }
    }, [user]);

    const handlePayNow = async () => {
        try {
            const res = await fetch(`http://localhost:5002/api/pay/${user.id}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            credentials: 'include'
            });

            const data = await res.json();
            if (res.ok) {
                alert('Payment successful!');
                setAmountDue(0);
            } 
            else {
                alert('Payment failed: ' + data.error);
            }
        } catch (err) {
            alert('Network error');
            console.error(err);
        }
    };


    useEffect(() => {
        if (!weeklyActivity.length) return;

        const ctx = document.getElementById('activityChart');
        if (ctx) {
            new Chart(ctx, {
            type: 'bar',
            data: {
                labels: weeklyActivity.map(item => item.day),
                datasets: [{
                label: 'Sessions',
                data: weeklyActivity.map(item => item.sessions),
                backgroundColor: 'rgba(67, 97, 238, 0.6)',
                borderRadius: 5
                }]
            },
            options: {
                responsive: true,
                scales: {
                y: {
                    beginAtZero: true,
                    ticks: { precision: 0 }
                }
                }
            }
            });
        }
    }, [weeklyActivity]);

    if (!user) return <p>Not Logged In...</p>;
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
                <div className="stat-value">{visitsLastMonth}</div>
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
                <div className="log-plate"><i className="fas fa-car"></i> {user.plate_number}</div>
                <div className="log-time">
                    {new Date(recentPayments[0].payment_time).toLocaleDateString('en-GB', {
                        day: 'numeric',
                        month: 'long',
                        year: 'numeric'
                    })}
                </div>
                <div className="log-status entry">{recentPayments[0].amount}</div>
            </div>
            <div className="log-item">
                <div className="log-plate"><i className="fas fa-car"></i> {user.plate_number}</div>
                <div className="log-time">
                    {new Date(recentPayments[1].payment_time).toLocaleDateString('en-GB', {
                        day: 'numeric',
                        month: 'long',
                        year: 'numeric'
                    })}
                </div>
                <div className="log-status entry">{recentPayments[1].amount}</div>
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
                <div className="stat-value" style={{ color: amountDue==0 ? '#2ecc71':'var(--danger)', fontSize: '2rem' }}>${amountDue}</div>
                <div className="stat-label">Current Balance</div>
                </div>
            </div>
            <button className="cta-button" style={{ width: '100%'}} onClick={handlePayNow}>
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
                <div className="stat-value" style={{ color: parkingStatus.status === 'Inside' ? '#2ecc71' : 'var(--danger)' }}>
                    {parkingStatus.status}
                </div>
                <div className="stat-label">Current Status</div>
                </div>
                {parkingStatus.status === 'Inside' && (
                <div className="stat-item">
                <div className="stat-value">{parkingSpot.spot}</div>
                <div className="stat-label">Parking Spot</div>
                </div>
                )}
                <div className="stat-item">
                <div className="stat-value">{new Date(parkingStatus.lastEntry).toLocaleTimeString()}</div>
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

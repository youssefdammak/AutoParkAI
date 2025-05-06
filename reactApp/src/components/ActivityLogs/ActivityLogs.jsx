import { useState, useEffect } from 'react';
import './ActivityLogs.css';

function ActivityLogs() {
    const [plates, setPlates] = useState([]);

    const fetchLatestPlate = async () => {
        try {
            const response = await fetch('http://localhost:5000/latest-plate');
            const data = await response.json();
            if (data.plate_number) {
                setPlates(prev => [...prev, { plate: data.plate_number, time: new Date().toLocaleTimeString() }]);
            }
        } catch (error) {
            console.error("Error fetching plate:", error);
        }
    };

    useEffect(() => {
        fetchLatestPlate(); // initial call

        const interval = setInterval(() => {
            fetchLatestPlate();
        }, 5000); // every 5 seconds

        return () => clearInterval(interval);
    }, [plates]); // include plates to avoid stale closure

    return (
        <section className="logs-container">
            <div className="logs-header">
                <h3 className="card-title"><i className="fas fa-list"></i> Recent Activity</h3>
                <div className="card-actions">
                    <button className="cta-button outline small" onClick={fetchLatestPlate}>
                        <i className="fas fa-sync-alt"></i> Refresh
                    </button>
                </div>
            </div>
            <div id="activityLogs">
                {plates.length > 0 ? (
                    plates.slice().reverse().map((entry, index) => (
                        <div className="log-item" key={index}>
                            <div className="log-plate"><i className="fas fa-car"></i> {entry.plate}</div>
                            <div className="log-time">{entry.time}</div>
                            <div className="log-status entry">Entry</div>
                        </div>
                    ))
                ) : (
                    <p>No plate detected yet.</p>
                )}
            </div>
        </section>
    );
}

export default ActivityLogs;

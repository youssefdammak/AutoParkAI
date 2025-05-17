import { useState, useEffect, useRef } from 'react';
import './ActivityLogs.css';

function ActivityLogs() {
    const [plates, setPlates] = useState(() => {
        const saved = localStorage.getItem("plates");
        return saved ? JSON.parse(saved) : [];
    });

    const latestPlatesRef = useRef([]); // useRef to persist between intervals

    useEffect(() => {
        latestPlatesRef.current = plates;
    }, [plates]);

    const fetchFromEndpoint = async (url) => {
        try {
            const response = await fetch(url);
            const data = await response.json();

            if (data.plate_number && data.status && data.entry_time) {
                return {
                    plate: data.plate_number,
                    time: data.entry_time,
                    status: data.status
                };
            }
        } catch (error) {
            console.error(`Error fetching from ${url}:`, error);
        }
        return null;
    };

    const fetchLatestPlates = async () => {
        const urls = [
            'http://localhost:5000/latest-plate',
            'http://localhost:5001/latest-plate'
        ];

        const results = await Promise.all(urls.map(fetchFromEndpoint));
        const newEntries = results.filter(entry => {
            return (
                entry &&
                !latestPlatesRef.current.some(
                    e => e.plate === entry.plate &&
                         e.status === entry.status &&
                         e.time === entry.time
                )
            );
        });

        if (newEntries.length > 0) {
            const updatedPlates = [...latestPlatesRef.current, ...newEntries];
            setPlates(updatedPlates);
            localStorage.setItem("plates", JSON.stringify(updatedPlates));
            window.dispatchEvent(new Event('platesUpdated'));
        }
    };

    useEffect(() => {
        fetchLatestPlates(); // initial fetch
        const interval = setInterval(fetchLatestPlates, 5000);
        return () => clearInterval(interval);
    }, []);

    return (
        <section className="logs-container">
            <div className="logs-header">
                <h3 className="card-title"><i className="fas fa-list"></i> Recent Activity</h3>
                <div className="card-actions">
                    <button className="cta-button outline small" onClick={fetchLatestPlates}>
                        <i className="fas fa-sync-alt"></i> Refresh
                    </button>
                </div>
            </div>
            <div id="activityLogs">
                {plates.length > 0 ? (
                    [...plates]
                        .sort((a, b) => new Date(b.time) - new Date(a.time))
                        .map((entry, index) => (
                            <div className="log-item" key={index}>
                                <div className="log-plate"><i className="fas fa-car"></i> {entry.plate}</div>
                                <div className="log-time">{entry.time}</div>
                                <div className={`log-status ${entry.status.toLowerCase() === 'exit' ? 'exit' : 'entry'}`}>
                                    {entry.status}
                                </div>
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

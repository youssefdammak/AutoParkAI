import './ActivityLogs.css'

function ActivityLogs(){
    return (
        <section className="logs-container">
            <div className="logs-header">
                <h3 className="card-title"><i className="fas fa-list"></i> Recent Activity</h3>
                <div className="card-actions">
                    <button className="cta-button outline small" onclick="refreshLogs()">
                        <i className="fas fa-sync-alt"></i> Refresh
                    </button>
                </div>
            </div>
            <div id="activityLogs">
                <div className="log-item">
                    <div className="log-plate">
                        <i className="fas fa-car"></i>
                        ABC1234
                    </div>
                    <div className="log-time">12:45 PM</div>
                    <div className="log-status entry">Entry</div>
                </div>
                <div className="log-item">
                    <div className="log-plate">
                        <i className="fas fa-car"></i>
                        XYZ7890
                    </div>
                    <div className="log-time">12:38 PM</div>
                    <div className="log-status exit">Exit</div>
                </div>
                <div className="log-item">
                    <div className="log-plate">
                        <i className="fas fa-car"></i>
                        DEF4567
                    </div>
                    <div className="log-time">12:25 PM</div>
                    <div className="log-status entry">Entry</div>
                </div>
                <div className="log-item">
                    <div className="log-plate">
                        <i className="fas fa-car"></i>
                        GHI8910
                    </div>
                    <div className="log-time">12:18 PM</div>
                    <div className="log-status exit">Exit</div>
                </div>
                <div className="log-item">
                    <div className="log-plate">
                        <i className="fas fa-car"></i>
                        JKL2345
                    </div>
                    <div className="log-time">12:05 PM</div>
                    <div className="log-status entry">Entry</div>
                </div>
            </div>
        </section>
    );
}

export default ActivityLogs;
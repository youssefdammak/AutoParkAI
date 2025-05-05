import './LiveFeed.css'

function LiveFeed(){
    return (
        <section className="live-feed-container">
            <div className="live-feed-header">
                <h3 className="card-title"><i className="fas fa-video"></i> Live Parking Feed</h3>
                <div className="card-actions">
                    <button className="cta-button outline">
                        <i className="fas fa-sync-alt"></i> Switch View
                    </button>
                </div>
            </div>
            <div className="live-feed">
                <img src="https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" alt="Parking lot view"/>
                <div className="live-feed-overlay">
                    <i className="fas fa-video"></i>
                    <p>Live Parking Lot View</p>
                </div>
            </div>
        </section>
    );
}

export default LiveFeed;
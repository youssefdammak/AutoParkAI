import './Hero.css';
function Hero(){
    return(
        <section className="hero">
            <h2>Revolutionary Smart Parking Solution</h2>
            <p>Experience seamless parking with our AI-powered system that guides you to available spots in real-time and helps you find your car effortlessly.</p>
            <div className="cta-buttons">
                <button className="cta-button">
                    <i className="fas fa-car"></i> Check Availability
                </button>
                <button className="cta-button outline">
                    <i className="fas fa-directions"></i> Get Directions
                </button>
            </div>
        </section>
    );
}
export default Hero;
import './Hero.css';
import { useNavigate } from 'react-router-dom';
function Hero(){
    const navigate = useNavigate();
    return(
        <section className="hero">
            <h2>Revolutionary Smart Parking Solution</h2>
            <p>Experience seamless parking with our AI-powered system that guides you to available spots in real-time and helps you find your car effortlessly.</p>
            
            <button className="cta-button" onClick={() => navigate('/map')}>
                <i className="fas fa-car"></i> Check Availability
            </button>
        </section>
    );
}
export default Hero;
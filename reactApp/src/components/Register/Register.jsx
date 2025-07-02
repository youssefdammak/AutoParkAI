import './Register.css'

function Register(){
    return(
        <div className='login-bg'>
            <div className='login-box'>
                <h2>Register</h2>
                <form action="">
                    <input type="text" placeholder="Username" required />
                    <input type="password" placeholder="Password" required />
                    <input type="password" placeholder="Re-Type Password" required />
                    <input type="text" placeholder="Car Plate Number" required />
                    <button type="submit">Sign Up</button>
                </form>
                <p className="signup-link">
                have an account? <a href="/account/Login">Login</a>
                </p>
            </div>
        </div>
    );
}
export default Register
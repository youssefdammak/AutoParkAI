import './Login.css'

function Login(){
    return(
        <div className='login-bg'>
            <div className='login-box'>
                <h2>Login</h2>
                <form action="http://localhost:5002/api/login" method='POST'>
                    <input type="text" name='username' placeholder="Username" required />
                    <input type="password" name='password' placeholder="Password" required />
                    <button type="submit">Log In</button>
                </form>
                <p className="signup-link">
                Don't have an account? <a href="/account/register">Create one</a>
                </p>
            </div>
        </div>
    );
}
export default Login
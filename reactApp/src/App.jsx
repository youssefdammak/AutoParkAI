import Header from './components/Header/Header.jsx'
import Hero from './components/Hero/Hero.jsx'
import Dashboard from './components/Dashboard/Dashboard.jsx'
import LiveFeed from './components/LiveFeed/LiveFeed.jsx'
import ActivityLogs from './components/ActivityLogs/ActivityLogs.jsx'
function App() {
  return(
    <div className='app'>
        <Header></Header>
        <div className="main-container">
            <Hero></Hero>
            <Dashboard></Dashboard>
            <LiveFeed></LiveFeed>
            <ActivityLogs></ActivityLogs>
        </div>
    </div>
  );
}

export default App

import "./App.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Projects from "./pages/Projects";
import About from "./pages/About";
import Navbar from "./Components/NavBar";
import ProjectDisplay from "./pages/ProjectDisplay";
import Footer from "./Components/Footer";

function App() {
  return (

    <div className="App">
      <Router>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/projects" element={<Projects />} />
          <Route path="/project/:id" element={<ProjectDisplay />} />

          <Route path="/about" element={<About />} />
        </Routes>
        <Footer />

      </Router>
    </div>
  );
}

export default App;

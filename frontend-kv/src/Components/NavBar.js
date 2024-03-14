import React, { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import "../styles/NavBar.css";
import { CgMenu } from "react-icons/cg";



function NavBar() {

  const [expandedNavbar, setExpandNavbar] = useState(false)
  const location = useLocation()
  useEffect(() => {
    setExpandNavbar(false);

  }, [location]);



  return (
    <nav className="sticky">
      <div className="navbar" id={expandedNavbar ? "open" : "close"}>
        <div className="container">
          <label className="logo"> </label>
          <button className="toggleButton"
            onClick={() => 
            {setExpandNavbar(prev => !prev)}}>
            <CgMenu />
          </button>
        </div>
        <div className="links">
          <ul>
            <li><Link to="/"> Home </Link></li>
            <li><Link to="/projects"> Projects </Link></li>
            <li><Link to="/about"> About </Link></li>
          </ul>
        </div>
      </div>
    </nav>
  );
}

export default NavBar;

import React from 'react'
import { FaLinkedin } from "react-icons/fa";
import { FaGithub } from "react-icons/fa";
import "../styles/Footer.css"

function Footer() {
  return (
  <div className="footer">
    <div className="footerIcon">
        <a href="https://www.linkedin.com/in/karina-verma-565027187/" target="_blank" rel="noreferrer">
        <FaLinkedin />
        </a>
        <a href="https://github.com/karinavermaa" target="_blank" rel="noreferrer">
        <FaGithub />
        </a>

    </div>
        <footerlogo> Karina Verma 2024</footerlogo>
    </div>
  )
}

export default Footer
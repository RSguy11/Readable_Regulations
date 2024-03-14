import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faGithub, faLinkedin } from '@fortawesome/free-brands-svg-icons';


// LinkedInButton.js
const LinkedInButton = ({ profileUrl }) => {
    const openLinkedInProfile = () => {
      window.open(profileUrl, '_blank');
    };
  
    return (
      <button className="social-button" onClick={openLinkedInProfile}>
        <FontAwesomeIcon icon={faLinkedin} size="2x" />
      </button>
    );
  };
  
  export default LinkedInButton;
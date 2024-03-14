import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faGithub, faLinkedin } from '@fortawesome/free-brands-svg-icons';


// GitHubButton.js
const GitHubButton = ({ profileUrl }) => {
    const openGitHubProfile = () => {
      window.open(profileUrl, '_blank');
    };
  
    return (
      <button className="social-button" onClick={openGitHubProfile}>
        <FontAwesomeIcon icon={faGithub} size="2x" />
      </button>
    );
  };
  
  export default GitHubButton;

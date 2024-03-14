import { useState, useEffect } from 'react';
import "../styles/Home.css";

const TypeText = ({ text, delay, infinite }) => {
  const [currentText, setCurrentText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    let timeout;

    if (currentIndex < text.length) {
      timeout = setTimeout(() => {
        setCurrentText(prevText => prevText + text[currentIndex]);
        setCurrentIndex(prevIndex => prevIndex + 1);
      }, delay);

    } else if (infinite) {
      // Animation complete, pause for a bit, then reset
      timeout = setTimeout(() => {
        setCurrentIndex(0);
        setCurrentText('');
      }, 2000); // Adjust the duration of the pause before resetting
    }

    return () => clearTimeout(timeout);
  }, [currentIndex, delay, infinite, text]);

  return <span>{currentText}</span>;
};

export default TypeText;
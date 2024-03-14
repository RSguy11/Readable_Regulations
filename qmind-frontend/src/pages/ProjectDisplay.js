import React from 'react'
import { useParams } from 'react-router-dom'
import { ProjectList } from '../Helpers/ProjectList';
import "../styles/ProjectDisplay.css"
function ProjectDisplay() {
    const { id }= useParams();
    // Where we access project
    const project = ProjectList[id];

  return (
    <div className="project">
        <h1> {project.name}</h1>
        <img src={project.image} alt=""/>
        
    </div>
  );
}

export default ProjectDisplay

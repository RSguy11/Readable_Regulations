import ProjectItems from "../Components/ProjectItems";
import website from '../Assets/website.png'
import "../styles/Projects.css";
import {ProjectList} from "../Helpers/ProjectList"

function Projects() {
    return (
        
      <div className="projects">
        <h1> Personal Projects </h1>
        <div className="projectList"></div>
        {/* <ProjectItems name="Personal Website" image={website}/> */}
        {ProjectList.map((project, index) =>{
            return <ProjectItems id={index} name={project.name} image={project.image}/>;
        })}

    </div>

    
    );
}

export default Projects;
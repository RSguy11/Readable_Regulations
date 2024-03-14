import TypeText from "../Components/TypeText";
import "../styles/Home.css";
import logo from '../Assets/pete.jpeg'

import FileUpload from "../Components/file_upload";
import { MdOutlineDriveFolderUpload } from "react-icons/md";






function Home() {
  const githubProfileUrl = 'https://github.com/your-username';
  const emailAddress = 'your-email@example.com';
  const linkedInProfileUrl = 'https://www.linkedin.com/in/your-linkedin-profile';

    
    return (
      <div className="home">
        <div className="homeHeader">
            <img src={logo} alt="" />



            <h3>
                 <TypeText text="Readable Regulations" delay={150} infinite />
            </h3>

            <div className="prompt">
          <p> Making regulations understandable and accessible. </p>

        </div>

      </div>
      <div>
    </div>

      <div className="upload">
        <h1> Upload</h1>
        <FileUpload />

      </div>
    </div>
  );
}


export default Home;
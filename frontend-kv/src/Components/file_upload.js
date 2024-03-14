import React, { useState } from 'react';
import { MdOutlineDriveFolderUpload } from 'react-icons/md';

const FileUpload = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);

  const handleFileChange = (event) => {
    const files = event.target.files;
    setSelectedFiles([...selectedFiles, ...files]);
  };

  return (
    <div style={{ position: 'relative' }}> {/* Set container's position to relative */}
      <label htmlFor="file-upload" style={{ cursor: 'pointer', position: 'absolute', top: 0, left: 0 }}> {/* Position the label absolutely */}
        <MdOutlineDriveFolderUpload size={32} />
      </label>
      <input
        id="file-upload"
        type="file"
        style={{ display: 'none' }}
        onChange={handleFileChange}
        multiple
      />
      {selectedFiles.length > 0 && (
        <div style={{ marginLeft: '40px' }}> {/* Adjust margin-left to maintain space for the icon */}
          <h3>Uploaded Files:</h3>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {selectedFiles.map((file, index) => (
              <li key={index}>{file.name}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default FileUpload;

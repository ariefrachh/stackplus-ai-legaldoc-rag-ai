import React from "react";

export default function FileList({ files, selectedFile, setSelectedFile }) {
  return (
    <div className="flex flex-wrap gap-2 mt-4">

      {files.map((f) => (
        <button
          key={f.filename}
          onClick={() => setSelectedFile(f.filename)}
          className={`px-3 py-1 rounded-lg text-sm transition 
          ${
            selectedFile === f.filename
              ? "bg-primary text-white"
              : "bg-gray-200 hover:bg-gray-300"
          }`}
        >
          {f.filename}
        </button>
      ))}

    </div>
  );
}
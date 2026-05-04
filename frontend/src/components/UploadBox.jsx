import { uploadFile } from "../api";
import React from "react";

export default function UploadBox({ setSelectedFile, reloadFiles }) {

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const res = await uploadFile(file);

    setSelectedFile(res.filename);
    reloadFiles();
  }

  return (
    <div className="bg-white p-6 rounded-xl shadow border text-center">

      <p className="mb-4 text-gray-600">
        Upload dokumen hukum (PDF)
      </p>

      <input
        type="file"
        accept="application/pdf"
        onChange={handleUpload}
        className="block w-full text-sm"
      />

    </div>
  );
}
import React, { useState, useEffect, useRef } from "react";
import { askQuestion, getFiles, uploadFile } from "../api";

export default function Hero() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileToUpload, setFileToUpload] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);

  const chatEndRef = useRef(null);

  // 🔥 LOAD FILE + AUTO REFRESH (SYNC QDRANT)
  useEffect(() => {
    loadFiles();

    const interval = setInterval(() => {
      loadFiles();
    }, 4000); // tiap 4 detik sync

    return () => clearInterval(interval);
  }, []);

  async function loadFiles() {
    try {
      const data = await getFiles();
      const newFiles = data.files || [];

      setFiles(newFiles);

      // 🔥 kalau file yg dipilih sudah tidak ada → reset
      if (
        selectedFile &&
        !newFiles.some((f) => f.filename === selectedFile)
      ) {
        setSelectedFile(null);
      }
    } catch (err) {
      console.error(err);
    }
  }

  async function handleUpload() {
    if (!fileToUpload) return;

    setUploading(true);

    try {
      const res = await uploadFile(fileToUpload);

      console.log("UPLOAD RESPONSE:", res);

      if (res.filename) {
        setSelectedFile(res.filename);
      }

      await loadFiles();

    } catch (err) {
      console.error("Upload error:", err);
      alert("Upload gagal");
    }

    setUploading(false);
    setFileToUpload(null);
  }

  // 🔥 SCROLL KE BAWAH OTOMATIS
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 🔥 SEND MESSAGE (ANTI DOUBLE ENTER)
  async function send() {
    if (!input.trim() || loading) return;

    if (!selectedFile) {
      alert("⚠️ Upload & pilih dokumen dulu");
      return;
    }

    const question = input;

    setMessages((prev) => [...prev, { text: question, sender: "user" }]);
    setInput("");
    setLoading(true);

    try {
      const res = await askQuestion(question, selectedFile);

      setMessages((prev) => [
        ...prev,
        { text: res.answer, sender: "bot", sources: res.sources },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { text: "❌ Server error", sender: "bot" },
      ]);
    }

    setLoading(false);
  }

  // 🔥 ENTER HANDLER (NO DOUBLE SEND)
  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();

      if (!loading) {
        send();
      }
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-6 py-16 grid md:grid-cols-2 gap-10">

      {/* LEFT */}
      <div>
        <p className="text-primary font-semibold mb-4">
          #1 AI-POWERED LEGAL PLATFORM  
        </p>

        <h1 className="text-4xl md:text-5xl font-bold leading-tight mb-6 text-black">
          Riset & Analisis <br />
          Dokumen Hukum <br />
          <span className="text-primary">
            Lebih Cepat & Cerdas
          </span>
        </h1>

        {/* Upload */}
        <div className="bg-white p-5 rounded-2xl shadow mb-4 max-w-xl">

          {/* Choose File */}
          <input
            type="file"
            onChange={(e) => {
              const file = e.target.files[0];
              if (file) setFileToUpload(file);
            }}
            className="mb-4 block w-full text-sm"
          />

          {/* Drag & Drop */}
          <div
            className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center text-gray-500 mb-4 hover:border-primary transition cursor-pointer"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const file = e.dataTransfer.files[0];
              if (file) setFileToUpload(file);
            }}
          >
            Drag & Drop file di sini
            {fileToUpload && (
              <div className="mt-2 text-sm text-gray-700 font-medium">
                {fileToUpload.name}
              </div>
            )}
          </div>

          {/* Upload Button */}
          <button
            onClick={handleUpload}
            disabled={!fileToUpload || uploading}
            className={`w-full py-3 rounded-xl text-white transition
            ${
              !fileToUpload || uploading
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-primary hover:opacity-90"
            }`}
          >
            {uploading ? "Uploading..." : "Upload"}
          </button>
        </div>

        {/* File List */}
        <div className="flex flex-wrap gap-2">
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
      </div>

      {/* RIGHT CHAT */}
      <div className="bg-white rounded-2xl p-6 flex flex-col h-[520px] shadow-[0_10px_40px_rgba(0,0,0,0.08)]">

        {/* LOADING UPLOAD */}
        {uploading && (
          <div className="text-sm text-blue-500 mb-2 flex items-center gap-2 animate-pulse">
            <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
            Dokumen sedang diproses...
          </div>
        )}

        {/* CHAT BODY */}
        <div className="flex-1 overflow-y-auto mb-4 space-y-3 pr-2">

          {messages.map((m, i) => (
            <div
              key={i}
              className={`flex ${
                m.sender === "user"
                  ? "justify-end"
                  : "justify-start"
              }`}
            >
              <div
                className={`px-4 py-3 rounded-2xl max-w-[75%] w-fit whitespace-pre-wrap break-words
                ${
                  m.sender === "user"
                    ? "bg-primary text-white"
                    : "bg-gray-100 text-gray-800"
                }`}
              >
                {m.text}

                {m.sources && (
                  <div className="text-xs mt-2 text-gray-500">
                    {m.sources.map((s, idx) => (
                      <div key={idx}>
                        Pasal {s.pasal_number}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="text-sm text-gray-400">
              AI sedang mengetik...
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* INPUT */}
        <div className="flex gap-2">
          <textarea
            className="flex-1 border rounded-xl p-3 resize-none focus:outline-none 
            transition-all duration-200 focus:ring-2 focus:ring-primary"
            rows={2}
            placeholder={
              input.length === 0
                ? "Tanyakan apapun terkait dokumen legal yang anda pilih"
                : ""
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />

          <button
            onClick={send}
            disabled={!input.trim() || !selectedFile || loading}
            className={`px-4 rounded-xl text-white transition
            ${
              !input.trim() || !selectedFile || loading
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-primary hover:opacity-90"
            }`}
          >
            Kirim
          </button>
        </div>

      </div>
    </div>
  );
}
const API_BASE =
  import.meta.env.VITE_API_URL ||
  "https://backend-legaldoc-stackplus-production.up.railway.app";

export async function askQuestion(question, selectedFile) {
  const res = await fetch(`${API_BASE}/query/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      filter_source: selectedFile,
      include_sources: true,
    }),
  });

  return res.json();
}

export async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/upload/`, {
    method: "POST",
    body: formData,
  });

  return res.json();
}

export async function getFiles() {
  const res = await fetch(`${API_BASE}/upload/files`);
  return res.json();
}
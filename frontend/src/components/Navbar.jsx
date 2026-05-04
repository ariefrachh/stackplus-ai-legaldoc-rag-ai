import React from "react";
import logo from "../assets/logo.png";

export default function Navbar() {
  return (
    <nav className="bg-white shadow-sm px-8 py-4 flex justify-between items-center">
      
      {/* LEFT */}
      <div className="flex items-center gap-3">
        <img src={logo} alt="StackPlus" className="h-14 w-auto rounded-xl object-cover" />
        <span className="text-xl font-semibold text-blue-700">
          Stackplus Legal AI
        </span>
      </div>

      {/* CENTER */}
      <div className="hidden md:flex gap-8 text-gray-600">
        <a href="#">Fitur</a>
        <a href="#">Harga</a>
        <a href="#">Kontak</a>
      </div>

      {/* RIGHT */}
      <div className="flex gap-3">
        <button className="px-5 py-2 border border-primary text-primary rounded-lg">
          Masuk
        </button>

        <button className="px-5 py-2 bg-primary text-white rounded-lg">
          Daftar
        </button>
      </div>
    </nav>
  );
}
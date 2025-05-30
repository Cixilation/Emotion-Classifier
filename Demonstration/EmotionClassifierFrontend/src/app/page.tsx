"use client";
import { useState, useRef } from "react";

export default function Home() {
  const [detectedEmotion, setDetectedEmotion] = useState("Are you Ready?");
  const [note, setNote] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState(null);

  const fileInputRef = useRef(null);
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);
  const streamRef = useRef(null);

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const handleFileUpload = async (file) => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);
    setNote(`${file.name || "Recording"} is uploaded.`);

    try {
      const res = await fetch("http://localhost:8080/classify-audio", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      console.log("Emotion:", data.emotion);
      setDetectedEmotion(data.emotion);
    } catch (err) {
      console.error("Upload failed", err);
      setNote("Upload failed.");
    }
  };

  const onFileInputChange = async (event) => {
    const file = event.target.files[0];
    await handleFileUpload(file);
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Try to use WAV format, fallback to webm
      let options = { mimeType: "audio/wav" };
      if (!MediaRecorder.isTypeSupported("audio/wav")) {
        options = { mimeType: "audio/webm" };
      }

      mediaRecorder.current = new MediaRecorder(stream, options);
      audioChunks.current = [];

      mediaRecorder.current.ondataavailable = (event) => {
        audioChunks.current.push(event.data);
      };

      mediaRecorder.current.onstop = async () => {
        const mimeType = mediaRecorder.current.mimeType;
        const audioBlob = new Blob(audioChunks.current, { type: mimeType });

        // Determine file extension based on mime type
        const extension = mimeType.includes("wav") ? "wav" : "webm";
        const filename = `recording.${extension}`;

        const formData = new FormData();
        formData.append("file", audioBlob, filename);

        const url = URL.createObjectURL(audioBlob);
        setAudioURL(url);

        try {
          const response = await fetch("http://localhost:8080/classify-audio", {
            method: "POST",
            body: formData,
          });
          const data = await response.json();
          setDetectedEmotion(data.emotion);
          setNote("Upload successful.");
        } catch (err) {
          console.error("Upload failed", err);
          setNote("Recording upload failed.");
        }

        audioChunks.current = [];
      };

      mediaRecorder.current.start();
      setIsRecording(true);
      setNote("Recording started...");
    } catch (err) {
      console.error("Could not start recording", err);
      setNote("Microphone not found or permission denied.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && mediaRecorder.current.state !== "inactive") {
      mediaRecorder.current.stop();
    }
    setIsRecording(false);
    streamRef.current?.getTracks().forEach((track) => track.stop());
  };

  return (
    <div
      className="bg-[#1f1416] min-h-screen text-white flex items-center justify-center"
      style={{ fontFamily: '"Space Grotesk", "Noto Sans", sans-serif' }}
    >
      <div className="w-full max-w-xl border border-[#e7b5be] rounded-md p-6 text-center">
        <h2 className="text-lg font-bold tracking-tight">
          Emotion Recognition
        </h2>
        <p className="mt-2 text-base font-normal">
          Define your emotion using our built in model.
        </p>
        <div className="mt-4 flex justify-center gap-4 flex-wrap">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            className="flex items-center gap-2 px-5 h-12 rounded-full bg-[#e7b5be] text-[#1f1416] font-bold cursor-pointer"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              fill="currentColor"
              viewBox="0 0 256 256"
            >
              <path d="M128,176a48.05,48.05,0,0,0,48-48V64a48,48,0,0,0-96,0v64A48.05,48.05,0,0,0,128,176ZM96,64a32,32,0,0,1,64,0v64a32,32,0,0,1-64,0Zm40,143.6V232a8,8,0,0,1-16,0V207.6A80.11,80.11,0,0,1,48,128a8,8,0,0,1,16,0,64,64,0,0,0,128,0,8,8,0,0,1,16,0A80.11,80.11,0,0,1,136,207.6Z"></path>
            </svg>
            {isRecording ? "Stop Recording" : "Record"}
          </button>

          {/* {audioURL && (
            <div className="mt-4">
              <audio controls src={audioURL}></audio>
            </div>
          )} */}

          <div>
            <input
              type="file"
              accept=".wav,.webm,.mp3"
              onChange={onFileInputChange}
              ref={fileInputRef}
              className="hidden"
              id="fileInput"
            />

            <button
              onClick={triggerFileInput}
              className="flex items-center gap-2 px-5 h-12 rounded-full bg-[#402b2f] text-white font-bold cursor-pointer"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                fill="currentColor"
                viewBox="0 0 256 256"
              >
                <path d="M240,136v64a16,16,0,0,1-16,16H32a16,16,0,0,1-16-16V136a16,16,0,0,1,16-16H80a8,8,0,0,1,0,16H32v64H224V136H176a8,8,0,0,1,0-16h48A16,16,0,0,1,240,136ZM85.66,77.66,120,43.31V128a8,8,0,0,0,16,0V43.31l34.34,34.35a8,8,0,0,0,11.32-11.32l-48-48a8,8,0,0,0-11.32,0l-48,48A8,8,0,0,0,85.66,77.66ZM200,168a12,12,0,1,0-12,12A12,12,0,0,0,200,168Z"></path>
              </svg>
              Upload File
            </button>
          </div>
        </div>
        {note && <p className="mt-2 text-sm text-gray-300 italic">{note}</p>}
        <div className="mt-6">
          <h3 className="mt-4 text-lg font-bold">Emotion Detected</h3>
          <p className="mt-1 text-2xl font-extrabold text-black bg-[#ffffff] px-4 py-2 rounded-full inline-block">
            {detectedEmotion}
          </p>
        </div>
      </div>
    </div>
  );
}

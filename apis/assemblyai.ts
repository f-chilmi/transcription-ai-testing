import { AssemblyAI, TranscribeParams } from "assemblyai";
import fs from "fs/promises";
import path from "path";

const client = new AssemblyAI({
  apiKey: "9bdae4331b39404994bc3d0dd8995638",
});

// change audio file here
const audioFile = "./assets/audio3.mp3";

// change the response path here
const responseJson = "assemblyai-nano-audio3-multichannel-detectlanguage.json";

const params: TranscribeParams = {
  audio: audioFile,
  speech_model: "nano",
  // language_detection: true,
  language_code: "ar",
  // multichannel: true,
};

// Required to handle __dirname in ESM
// const __dirname = path.dirname(__filename);

export const runAssemblyAI = async () => {
  const transcript = await client.transcripts.transcribe(params);

  const filePath = path.join("response", responseJson);
  await fs.writeFile(filePath, JSON.stringify(transcript, null, 2), "utf-8");

  for (const utterance of transcript.utterances!) {
    console.log(`Channel ${utterance.speaker}: ${utterance.text}`);
  }
};

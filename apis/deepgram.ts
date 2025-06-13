import { createClient } from "@deepgram/sdk";
import fs from "fs";
import asyncFs from "fs/promises";
import path from "path";

// change audio file here
const audioFile = "./assets/audio1.mp3";

// change the response path here
const responseJson = "deepgram-whisperlarge-audio1.json";

export const runDeepgram = async () => {
  const deepgramApiKey = "8ac6afa898aef089cd9d072c20e49a618f4dc852";

  const deepgram = createClient(deepgramApiKey);

  const { result, error } = await deepgram.listen.prerecorded.transcribeFile(
    fs.readFileSync(audioFile),
    {
      model: "whisper-large",
      language: "en",
      smart_format: true,
      diarize: true,
      multichannel: true,
      summarize: true,
      topics: true,
      punctuate: true,
      utterances: true,
    }
  );

  if (error) {
    console.error(error);
  } else {
    const filePath = path.join("response", responseJson);
    await asyncFs.writeFile(filePath, JSON.stringify(result, null, 2), "utf-8");

    console.dir(result, { depth: null });
  }
};

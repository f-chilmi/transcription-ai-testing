import * as fs from "fs";
import * as path from "path";

interface Word {
  word: string;
  start: number;
  end: number;
  score: number;
  speaker: string;
}

interface Segment {
  start: number;
  end: number;
  text: string;
  words: Word[];
  speaker: string;
}

interface WhisperXResult {
  result: {
    segments: Segment[];
  };
}

interface FormattedOutput {
  speakers: {
    [speakerName: string]: string;
  };
}

const fileName = "whisper-audio1.json";

export function formatWhisperXJson(): void {
  try {
    const responseDir = path.join(process.cwd(), "response");
    let filePath: string;

    if (fileName) {
      // If fileName provided, check different variations
      const possiblePaths = [
        path.join(responseDir, fileName), // exact filename
        path.join(responseDir, `${fileName}.json`), // add .json extension
        fileName.startsWith(responseDir)
          ? fileName
          : path.join(responseDir, fileName), // full path check
      ];

      let foundPath: string | null = null;
      for (const testPath of possiblePaths) {
        if (fs.existsSync(testPath)) {
          foundPath = testPath;
          break;
        }
      }

      if (!foundPath) {
        console.log(`❌ File not found: ${fileName}`);
        console.log(`🔍 Searched in:`);
        possiblePaths.forEach((p) => console.log(`   - ${p}`));
        return;
      }

      filePath = foundPath;
      console.log(`📄 Processing: ${path.basename(filePath)}`);
    } else {
      // Auto-detect WhisperX JSON files in response directory
      if (!fs.existsSync(responseDir)) {
        console.log(`❌ Response directory not found: ${responseDir}`);
        return;
      }

      const files = fs.readdirSync(responseDir);
      const whisperFiles = files.filter(
        (file) =>
          file.includes("whisper") &&
          file.endsWith(".json") &&
          !file.includes("-formatted")
      );

      if (whisperFiles.length === 0) {
        console.log("❌ No WhisperX JSON files found in response directory");
        console.log(
          `📁 Available files in response/:`,
          files.filter((f) => f.endsWith(".json"))
        );
        return;
      }

      // Use the most recent file if multiple exist
      const fullPaths = whisperFiles.map((file) => ({
        name: file,
        path: path.join(responseDir, file),
        stat: fs.statSync(path.join(responseDir, file)),
      }));

      fullPaths.sort((a, b) => b.stat.mtime.getTime() - a.stat.mtime.getTime());
      filePath = fullPaths[0].path;

      console.log(`📄 Auto-detected and processing: ${fullPaths[0].name}`);
    }

    // Read the input JSON file
    const jsonContent = fs.readFileSync(filePath, "utf8");
    const data: WhisperXResult = JSON.parse(jsonContent);

    // Sort segments by start time to maintain chronological order
    const sortedSegments = data.result.segments.sort(
      (a, b) => a.start - b.start
    );

    // Create two different formats

    // Format 1: Grouped by speaker (original request)
    const speakerTexts: { [speaker: string]: string[] } = {};
    sortedSegments.forEach((segment) => {
      const speaker = segment.speaker || "UNKNOWN_SPEAKER";

      if (!speakerTexts[speaker]) {
        speakerTexts[speaker] = [];
      }

      if (segment.text && segment.text.trim()) {
        speakerTexts[speaker].push(segment.text.trim());
      }
    });

    // Format 2: Chronological conversation flow
    const conversationFlow: Array<{
      speaker: string;
      text: string;
      timestamp: string;
    }> = [];
    sortedSegments.forEach((segment) => {
      if (segment.text && segment.text.trim()) {
        const speaker = segment.speaker || "UNKNOWN_SPEAKER";
        const speakerNumber = speaker.replace("SPEAKER_", "");
        const speakerLabel = `Speaker ${speakerNumber}`;

        // Format timestamp
        const minutes = Math.floor(segment.start / 60);
        const seconds = Math.floor(segment.start % 60);
        const timestamp = `${minutes}:${seconds.toString().padStart(2, "0")}`;

        conversationFlow.push({
          speaker: speakerLabel,
          text: segment.text.trim(),
          timestamp: timestamp,
        });
      }
    });

    // Create formatted output with both formats
    const formattedOutput = {
      // Grouped by speaker format
      speakers: {} as { [speakerName: string]: string },

      // Chronological conversation format
      conversation: conversationFlow,
    };

    // Convert speaker keys to more readable format and join texts
    Object.keys(speakerTexts)
      .sort()
      .forEach((speaker) => {
        const speakerNumber = speaker.replace("SPEAKER_", "");
        const speakerLabel = `Speaker ${speakerNumber}`;

        // Join all text segments for this speaker with spaces
        formattedOutput.speakers[speakerLabel] =
          speakerTexts[speaker].join(" ");
      });

    // Generate output file name in response directory
    const inputDir = path.dirname(filePath);
    const inputBaseName = path.basename(filePath, path.extname(filePath));
    const outputFilePath = path.join(
      inputDir,
      `${inputBaseName}-formatted.json`
    );

    // Write the formatted JSON
    fs.writeFileSync(
      outputFilePath,
      JSON.stringify(formattedOutput, null, 2),
      "utf8"
    );

    console.log(
      `✅ Successfully formatted and saved to: ${path.basename(outputFilePath)}`
    );

    // Also create a readable text version with conversation flow
    const textOutputPath = path.join(
      inputDir,
      `${inputBaseName}-formatted.txt`
    );
    let textOutput = "";

    // Add grouped format
    textOutput += "=== GROUPED BY SPEAKER ===\n\n";
    Object.entries(formattedOutput.speakers).forEach(([speaker, text]) => {
      textOutput += `${speaker}\n${text}\n\n`;
    });

    // Add conversation flow format
    textOutput += "\n=== CONVERSATION FLOW ===\n\n";
    conversationFlow.forEach((item) => {
      textOutput += `[${item.timestamp}] ${item.speaker}\n${item.text}\n\n`;
    });

    fs.writeFileSync(textOutputPath, textOutput.trim(), "utf8");
    console.log(
      `📄 Also created readable text version: ${path.basename(textOutputPath)}`
    );

    // Log summary
    const speakerCount = Object.keys(formattedOutput.speakers).length;
    const segmentCount = conversationFlow.length;
    console.log(
      `🎤 Found ${speakerCount} speaker(s) with ${segmentCount} segments in chronological order`
    );
  } catch (error) {
    console.error("❌ Error processing WhisperX file:", error);

    if (error instanceof SyntaxError) {
      console.error("Invalid JSON format in input file");
    } else if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      console.error("Input file not found");
    } else {
      console.error("Unexpected error occurred");
    }
  }
}

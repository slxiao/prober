
/**
 * 
 */

/**
 * @author Ahmed Metwally
 *
 */

import java.io.File;
import java.io.IOException;
import java.util.List;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

import de.siegmar.fastcsv.writer.CsvWriter;
import de.siegmar.fastcsv.writer.CsvAppender;

import java.nio.charset.StandardCharsets;

import java.io.BufferedReader;
import java.util.HashMap;

public class Main {

	public static void main(String[] args) throws IOException {
		ArgumentParser parser = ArgumentParsers.newFor("Main").build().defaultHelp(true)
				.description("get code metrics from source code");
		parser.addArgument("-t", "--type").choices("file", "directory").setDefault("file")
				.help("granularity of metrics, default is file");
		parser.addArgument("-o", "--output").setDefault("metrics.csv")
				.help("Output of code metrics csv file, default is metrics.csv");
		parser.addArgument("dirname").help("directory for metrics collection");

		Namespace ns = null;
		try {
			ns = parser.parseArgs(args);
		} catch (ArgumentParserException e) {
			parser.handleError(e);
			System.exit(1);
		}

		// get the Directory name from the user
		String DirName = ns.getString("dirname");
		System.out.println("Directory Name is: " + DirName);

		String Granularity = ns.getString("type");
		System.out.println("Granularity is: " + Granularity);

		// retrieve all .java files in the directory and subdirectories.
		List<String> JavaFiles = Utils.retrieveFiles(DirName);

		// parse files in a directory to list of char array
		List<char[]> FilesRead = Utils.ParseFilesInDir(JavaFiles);
		List<BufferedReader> FileReaders = Utils.GetReadersInDir(JavaFiles);

		HalsteadVistor halsteadVistorFile;
		McCabeVistor mccabeVistorFile;

		// Construct the AST of each java file. visit different nodes to get the number
		// of operors and operands
		// Retrieve the number of distinct operators, distinct operands,
		// total operators, and total operands for each .java file in the directory.
		// Sum each parameter from different files together to be used in Halstead
		// Complexity metrics.
		File file = new File(ns.getString("output"));
		CsvWriter csvWriter = new CsvWriter();

		CsvAppender csvAppender = csvWriter.append(file, StandardCharsets.UTF_8);

		csvAppender.appendLine("filepath", "vocabulary", "proglen", "calcprogLen", "volume", "difficulty", "effort",
				"timeteqprog", "timedelbugs", "distoprt", "distoper", "totoprt", "totoper", "cc", "linecount", "commentcount",
				"blankcount", "linecommentcount");

		for (int i = 0; i < FilesRead.size(); i++) {
			int OperatorCount = 0;
			int OperandCount = 0;
			int CC = 0;

			csvAppender.appendField(JavaFiles.get(i));
			halsteadVistorFile = (HalsteadVistor) Parser.parse(FilesRead.get(i), new HalsteadVistor());
			mccabeVistorFile = (McCabeVistor) Parser.parse(FilesRead.get(i), new McCabeVistor());
			HashMap<String, Integer> counts = new HashMap<String, Integer>();
			counts = LineCounter.getNumberOfLines(FileReaders.get(i));

			CC = mccabeVistorFile.cc;

			for (int f : halsteadVistorFile.oprt.values()) {
				OperatorCount += f;
			}

			for (int f : halsteadVistorFile.names.values()) {
				OperandCount += f;
			}

			HalsteadMetrics hal = new HalsteadMetrics();

			hal.setParameters(halsteadVistorFile.oprt.size(), halsteadVistorFile.names.size(), OperatorCount, OperandCount);
			csvAppender.appendField(Integer.toString(hal.getVocabulary()));
			csvAppender.appendField(Integer.toString(hal.getProglen()));
			csvAppender.appendField(String.format("%.2f", hal.getCalcProgLen()));
			csvAppender.appendField(String.format("%.2f", hal.getVolume()));
			csvAppender.appendField(String.format("%.2f", hal.getDifficulty()));
			csvAppender.appendField(String.format("%.2f", hal.getEffort()));
			csvAppender.appendField(String.format("%.2f", hal.getTimeReqProg()));
			csvAppender.appendField(String.format("%.2f", hal.getTimeDelBugs()));
			csvAppender.appendField(Integer.toString(halsteadVistorFile.oprt.size()));
			csvAppender.appendField(Integer.toString(halsteadVistorFile.names.size()));
			csvAppender.appendField(Integer.toString(OperatorCount));
			csvAppender.appendField(Integer.toString(OperandCount));
			csvAppender.appendField(Integer.toString(CC));
			csvAppender.appendField(Integer.toString(counts.get("lineCount")));
			csvAppender.appendField(Integer.toString(counts.get("commentCount")));
			csvAppender.appendField(Integer.toString(counts.get("blankCount")));
			csvAppender.appendField(Integer.toString(counts.get("linecommentcount")));
			csvAppender.endLine();
		}

		csvAppender.close();

	}
}

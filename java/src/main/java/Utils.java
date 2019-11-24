import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Utils {
  // parse file in char array
  public static char[] ReadFileToCharArray(String filePath) throws IOException {
    StringBuilder fileData = new StringBuilder(1000);
    BufferedReader reader = new BufferedReader(new FileReader(filePath));
    char[] buf = new char[10];
    int numRead = 0;
    while ((numRead = reader.read(buf)) != -1) {
      String readData = String.valueOf(buf, 0, numRead);
      fileData.append(readData);
      buf = new char[1024];
    }
    reader.close();

    return fileData.toString().toCharArray();
  }

  // parse files in a directory to list of char array
  public static List<char[]> ParseFilesInDir(List<String> JavaFiles) throws IOException {
    if (JavaFiles.isEmpty()) {
      System.out.println("There is no java source code in the provided directory");
      System.exit(0);
    }

    List<char[]> FilesRead = new ArrayList<char[]>();

    for (int i = 0; i < JavaFiles.size(); i++) {
      System.out.println("Now, reading: " + JavaFiles.get(i));
      FilesRead.add(ReadFileToCharArray(JavaFiles.get(i)));
    }

    return FilesRead;
  }

  // read files in a directory to list of readers
  public static List<BufferedReader> GetReadersInDir(List<String> JavaFiles) throws IOException {
    if (JavaFiles.isEmpty()) {
      System.out.println("There is no java source code in the provided directory");
      System.exit(0);
    }

    List<BufferedReader> FilesReaders = new ArrayList<BufferedReader>();

    for (int i = 0; i < JavaFiles.size(); i++) {
      System.out.println("Now, reading: " + JavaFiles.get(i));
      BufferedReader reader = new BufferedReader(new FileReader(JavaFiles.get(i)));
      FilesReaders.add(reader);
    }

    return FilesReaders;
  }


  // retrieve all .java files in the directory and subdirectories.
  public static List<String> retrieveFiles(String directory) {
    List<String> Files = new ArrayList<String>();
    File dir = new File(directory);

    if (!dir.isDirectory()) {
      System.out.println("The provided path is not a valid directory");
      System.exit(1);
    }

    for (File file : dir.listFiles()) {
      if (file.isDirectory()) {
        Files.addAll(retrieveFiles(file.getAbsolutePath()));
      }
      if (file.getName().endsWith((".java"))) {
        Files.add(file.getAbsolutePath());
      }
    }

    return Files;
  }

}

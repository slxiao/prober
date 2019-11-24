import org.eclipse.jdt.core.dom.AST;
import org.eclipse.jdt.core.dom.ASTParser;
import org.eclipse.jdt.core.dom.CompilationUnit;
import org.eclipse.jdt.core.dom.ASTVisitor;

public class Parser {
  // construct AST of the .java files
  public static ASTVisitor parse(char[] str, ASTVisitor visitor) {
    ASTParser parser = ASTParser.newParser(AST.JLS3);

    parser.setSource(str);
    parser.setKind(ASTParser.K_COMPILATION_UNIT);
    parser.setResolveBindings(true);
    final CompilationUnit cu = (CompilationUnit) parser.createAST(null);

    // visit nodes of the constructed AST
    // HalsteadVistor visitor = new HalsteadVistor();
    cu.accept(visitor);

    return visitor;
  }

}
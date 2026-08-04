import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/** Export one solved-frame catalog over every geometry boundary for entity remapping. */
public class ComsolBoundaryCatalogExport {
  static String f(double value) {
    return Double.isFinite(value) ? String.format(Locale.US, "%.17g", value) : "";
  }

  static double[] global(Model model, String dataset, String expression) {
    String tag = "global_" + Math.abs(expression.hashCode());
    model.result().numerical().create(tag, "EvalGlobal");
    model.result().numerical(tag).set("data", dataset);
    model.result().numerical(tag).set("expr", new String[] {expression});
    return model.result().numerical(tag).getReal()[0];
  }

  public static void main(String[] args) throws Exception {
    Locale.setDefault(Locale.US);
    if (args.length < 3) {
      throw new IllegalArgumentException("usage: solved.mph output.csv dataset");
    }
    Path mph = Paths.get(args[0]).toAbsolutePath();
    Path output = Paths.get(args[1]).toAbsolutePath();
    String dataset = args[2];
    ModelUtil.initStandalone(false);
    Model model = ModelUtil.load("boundary_catalog", mph.toString());
    int[] entityCounts = model.geom("geom1").measureFinal().getNEntities();
    int boundaryCount = entityCounts[1];
    int lastSolution = global(model, dataset, "t").length;
    Files.createDirectories(output.getParent());
    try (PrintWriter writer =
        new PrintWriter(
            new BufferedWriter(new FileWriter(output.toFile(), StandardCharsets.UTF_8)))) {
      writer.println("boundary_entity,node,R_m,Z_m");
      for (int boundary = 1; boundary <= boundaryCount; boundary++) {
        String tag = "catalog_" + boundary;
        model.result().numerical().create(tag, "Eval");
        model.result().numerical(tag).set("data", dataset);
        model.result().numerical(tag).selection().geom("geom1", 1);
        model.result().numerical(tag).selection().set(boundary);
        model.result().numerical(tag)
            .set("expr", new String[] {"R/1[m]", "Z/1[m]"});
        model.result().numerical(tag).set("solnum", lastSolution);
        double[][] data = model.result().numerical(tag).getReal();
        for (int row = 0; row < data.length; row++) {
          ArrayList<String> values = new ArrayList<>();
          values.add(Integer.toString(boundary));
          values.add(Integer.toString(row));
          for (double value : data[row]) values.add(f(value));
          writer.println(String.join(",", values));
        }
      }
    }
    System.out.println(
        "boundary catalog boundaries="
            + boundaryCount
            + " entityCounts="
            + Arrays.toString(entityCounts)
            + " output="
            + output);
  }
}

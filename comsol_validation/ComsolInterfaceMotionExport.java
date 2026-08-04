import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/** Export COMSOL structural acceleration at frozen native-interface quadrature points. */
public class ComsolInterfaceMotionExport {
  static String format(double value) {
    return Double.isFinite(value) ? String.format(Locale.US, "%.17g", value) : "";
  }

  static double[] global(Model model, String dataset, String expression) {
    String tag = "global_" + Math.abs(expression.hashCode());
    model.result().numerical().create(tag, "EvalGlobal");
    model.result().numerical(tag).set("data", dataset);
    model.result().numerical(tag).set("expr", new String[]{expression});
    return model.result().numerical(tag).getReal()[0];
  }

  public static void main(String[] args) throws Exception {
    Locale.setDefault(Locale.US);
    if (args.length < 3) {
      throw new IllegalArgumentException(
          "usage: solved.mph interface_quadrature.csv output.csv [dataset]");
    }
    Path mph = Paths.get(args[0]).toAbsolutePath();
    Path samples = Paths.get(args[1]).toAbsolutePath();
    Path output = Paths.get(args[2]).toAbsolutePath();
    String dataset = args.length > 3 ? args[3] : "dset3";
    List<String> lines = Files.readAllLines(samples, StandardCharsets.UTF_8);
    TreeSet<Integer> boundaries = new TreeSet<>();
    for (int index = 0; index < lines.size() - 1; index++) {
      String[] columns = lines.get(index + 1).split(",");
      boundaries.add(Integer.parseInt(columns[12]));
    }

    ModelUtil.initStandalone(false);
    Model model = ModelUtil.load("interface_motion_export", mph.toString());
    double f0 = model.param().evaluate("f0");
    double period = 1.0 / f0;
    double[] allTimes = global(model, dataset, "t");
    TreeMap<Double, Integer> selectedSolutions = new TreeMap<>();
    for (int index = 0; index < allTimes.length; index++) {
      if (allTimes[index] >= 3.0 * period - 1e-10) {
        selectedSolutions.put(allTimes[index], index + 1);
      }
    }
    Files.createDirectories(output.getParent());
    try (PrintWriter writer =
        new PrintWriter(
            new BufferedWriter(
                new FileWriter(output.toFile(), StandardCharsets.UTF_8)))) {
      writer.println(
          "time_s,time_index,boundary_entity,boundary_node_index,R_m,Z_m,"
              + "up_displacement_r_m,up_displacement_z_m,"
              + "down_displacement_r_m,down_displacement_z_m");
      for (int boundary : boundaries) {
        String tag = "boundary_motion_" + boundary;
        model.result().numerical().create(tag, "Eval");
        model.result().numerical(tag).set("data", dataset);
        model.result().numerical(tag).selection().geom("geom1", 1);
        model.result().numerical(tag).selection().set(boundary);
        model.result().numerical(tag)
            .set(
                "expr",
                new String[]{
                  "R/1[m]", "Z/1[m]", "up(u)", "up(w)", "down(u)", "down(w)"
                });
        int timeIndex = 0;
        for (Map.Entry<Double, Integer> solution : selectedSolutions.entrySet()) {
          model.result().numerical(tag).set("solnum", solution.getValue());
          double[][] data = model.result().numerical(tag).getReal();
          for (int node = 0; node < data.length; node++) {
            if (data[node].length != 6) {
              throw new RuntimeException(
                  "unexpected Eval shape boundary="
                      + boundary
                      + " rows="
                      + data.length
                      + " columns="
                      + data[node].length);
            }
            writer.println(
                String.join(
                    ",",
                    format(solution.getKey()),
                    Integer.toString(timeIndex),
                    Integer.toString(boundary),
                    Integer.toString(node),
                    format(data[node][0]),
                    format(data[node][1]),
                    format(data[node][2]),
                    format(data[node][3]),
                    format(data[node][4]),
                    format(data[node][5])));
          }
          timeIndex++;
        }
      }
    }
    System.out.println(
        "interface motion export complete: boundaries="
            + boundaries.size()
            + " times="
            + selectedSolutions.size()
            + " output="
            + output);
  }
}

import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/** Export pressure and outward radial pressure gradient at the physical/PML interface. */
public class ComsolAcousticInterfaceExport {
  static String format(double value) {
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
    try {
      run(args);
    } catch (Throwable error) {
      error.printStackTrace(System.err);
      throw error;
    }
  }

  static void run(String[] args) throws Exception {
    Locale.setDefault(Locale.US);
    Path mph =
        Paths.get(
                args != null && args.length > 0
                    ? args[0]
                    : "D:\\loudspeakerFEM_comsol_validation\\refined_mesh\\solved.mph")
            .toAbsolutePath();
    Path output =
        Paths.get(
                args != null && args.length > 1
                    ? args[1]
                    : "D:\\loudspeakerFEM_comsol_validation\\refined_mesh\\acoustic_interface_timeseries.csv")
            .toAbsolutePath();
    String dataset = args != null && args.length > 2 ? args[2] : "dset3";
    int[] boundaries = new int[] {79, 82, 92};

    ModelUtil.initStandalone(false);
    Model model = ModelUtil.load("acoustic_interface_export", mph.toString());
    double f0 = model.param().evaluate("f0");
    double period = 1.0 / f0;
    double[] allTimes = global(model, dataset, "t");
    TreeMap<Double, Integer> selected = new TreeMap<>();
    for (int index = 0; index < allTimes.length; index++) {
      if (allTimes[index] >= 3.0 * period - 1e-10) {
        selected.put(allTimes[index], index + 1);
      }
    }

    Files.createDirectories(output.getParent());
    try (PrintWriter writer =
        new PrintWriter(
            new BufferedWriter(new FileWriter(output.toFile(), StandardCharsets.UTF_8)))) {
      writer.println(
          "time_s,time_index,boundary_entity,node,R_m,Z_m,"
              + "p_Pa,dpdn_Pa_per_m,up_p_Pa,down_p_Pa,"
              + "up_dpdn_Pa_per_m,down_dpdn_Pa_per_m");
      for (int boundary : boundaries) {
        String tag = "acoustic_interface_" + boundary;
        model.result().numerical().create(tag, "Eval");
        model.result().numerical(tag).set("data", dataset);
        model.result().numerical(tag).selection().geom("geom1", 1);
        model.result().numerical(tag).selection().set(boundary);
        model.result().numerical(tag)
            .set(
                "expr",
                new String[] {
                  "R/1[m]",
                  "Z/1[m]",
                  "actd.p_t/1[Pa]",
                  "(R*d(actd.p_t,R)+Z*d(actd.p_t,Z))/(0.115[m]*1[Pa/m])",
                  "up(actd.p_t)/1[Pa]",
                  "down(actd.p_t)/1[Pa]",
                  "(R*up(d(actd.p_t,R))+Z*up(d(actd.p_t,Z)))/(0.115[m]*1[Pa/m])",
                  "(R*down(d(actd.p_t,R))+Z*down(d(actd.p_t,Z)))/(0.115[m]*1[Pa/m])"
                });
        int timeIndex = 0;
        for (Map.Entry<Double, Integer> solution : selected.entrySet()) {
          model.result().numerical(tag).set("solnum", solution.getValue());
          double[][] data = model.result().numerical(tag).getReal();
          for (int node = 0; node < data.length; node++) {
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
                    format(data[node][5]),
                    format(data[node][6]),
                    format(data[node][7])));
          }
          timeIndex++;
        }
      }
    }
    System.out.println(
        "acoustic interface export complete: times=" + selected.size() + " output=" + output);
  }
}

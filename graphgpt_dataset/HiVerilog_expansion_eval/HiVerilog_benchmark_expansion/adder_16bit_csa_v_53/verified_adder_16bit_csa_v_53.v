`timescale 1ns / 1ns

module comparator_16bit(a, b, eq, gt, lt);
input [15:0] a, b;
output eq, gt, lt;

wire [3:0] eq_local, gt_local, lt_local;

comparator_4bit_slice comp_slice1(
.a(a[3:0]),
.b(b[3:0]),
.eq(eq_local[0]),
.gt(gt_local[0]),
.lt(lt_local[0]));

comparator_4bit_slice comp_slice2(
.a(a[7:4]),
.b(b[7:4]),
.eq(eq_local[1]),
.gt(gt_local[1]),
.lt(lt_local[1]));

comparator_4bit_slice comp_slice3(
.a(a[11:8]),
.b(b[11:8]),
.eq(eq_local[2]),
.gt(gt_local[2]),
.lt(lt_local[2]));

comparator_4bit_slice comp_slice4(
.a(a[15:12]),
.b(b[15:12]),
.eq(eq_local[3]),
.gt(gt_local[3]),
.lt(lt_local[3]));

assign eq = &eq_local;
assign gt = |(gt_local & ~(|lt_local));
assign lt = |(lt_local & ~(|gt_local));
endmodule

module comparator_4bit_slice(a, b, eq, gt, lt);
input [3:0] a, b;
output eq, gt, lt;

wire [3:0] eq_bit, gt_bit, lt_bit;

genvar i;
generate
    for (i = 0; i < 4; i = i + 1) begin : comp_bits
        assign eq_bit[i] = (a[i] == b[i]);
        assign gt_bit[i] = (a[i] & ~b[i]);
        assign lt_bit[i] = (~a[i] & b[i]);
    end
endgenerate

assign eq = &eq_bit;
assign gt = |(gt_bit & ~(|lt_bit));
assign lt = |(lt_bit & ~(|gt_bit));
endmodule

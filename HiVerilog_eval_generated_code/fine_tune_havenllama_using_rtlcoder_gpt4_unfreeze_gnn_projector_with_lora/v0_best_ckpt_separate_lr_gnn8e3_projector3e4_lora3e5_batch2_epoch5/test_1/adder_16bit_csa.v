module adder_16bit_csa(a, b, cin, sum, cout);
input [15:0] a, b;
input cin;
output [15:0] sum;
output cout;

wire [3:0] c1, c2, c3, c4;
wire [15:0] s1, s2, s3, s4;
wire c1_out, c2_out, c3_out, c4_out;



ripple_carry_adder ripple_carry_adder1(a[3:0], b[3:0], cin, c1);


carry_select_adder carry_select_adder1(a[7:4], b[7:4], c1, s1, c1_out);
carry_select_adder carry_select_adder2(a[11:8], b[11:8], c1_out, s2, c2_out);
carry_select_adder carry_select_adder3(a[15:12], b[15:12], c2_out, s3, c3_out);
carry_select_adder carry_select_adder4(a[15:12], b[15:12], c3_out, s4, c4_out);

assign sum = s1 + s2 + s3 + s4;
assign cout = c1_out | c2_out | c3_out | c4_out;

endmodule


module ripple_carry_adder(a, b, cin, cout);
input [3:0] a, b;
input cin;
output cout;

wire [3:0] s;
wire c_out;

assign s = a + b + cin;
assign cout = c_out;

endmodule


module carry_select_adder(a, b, cin, sum, cout);
input [3:0] a, b;
input cin;
output [3:0] sum;
output cout;

wire [3:0] s;
wire c_out;

assign s = a + b + cin;
assign sum = s;
assign cout = c_out;

endmodule
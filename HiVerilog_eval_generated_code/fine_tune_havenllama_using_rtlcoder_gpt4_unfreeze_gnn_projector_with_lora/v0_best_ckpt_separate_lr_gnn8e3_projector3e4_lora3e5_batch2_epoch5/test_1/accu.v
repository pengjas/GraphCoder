module accu (
    input               clk,
    input               rst_n,
    input       [7:0]   data_in,
    input               valid_in,

    output              valid_out,
    output      [9:0]   data_out
);

reg                  valid_out;
reg     [9:0]   data_out;

wire                valid_in_d1;
wire                valid_in_d2;
endmodule
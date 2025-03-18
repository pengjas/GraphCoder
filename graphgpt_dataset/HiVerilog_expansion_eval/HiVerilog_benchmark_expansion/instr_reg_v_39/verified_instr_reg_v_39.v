module data_mux (
    input clk,
    input [2:0] sel,
    input [7:0] d0,
    input [7:0] d1,
    input [7:0] d2,
    output [7:0] data_out
);

    // Submodule for data paths
    wire [7:0] path0_out, path1_out, path2_out;

    mux_path mux0 (
        .clk(clk),
        .sel(sel),
        .data(d0),
        .path_out(path0_out),
        .sel_code(3'b000) // Path 0 selection code
    );

    mux_path mux1 (
        .clk(clk),
        .sel(sel),
        .data(d1),
        .path_out(path1_out),
        .sel_code(3'b001) // Path 1 selection code
    );

    mux_path mux2 (
        .clk(clk),
        .sel(sel),
        .data(d2),
        .path_out(path2_out),
        .sel_code(3'b010) // Path 2 selection code
    );

    // Output assignment using a multiplexer logic
    assign data_out = (sel == 3'b000) ? path0_out :
                      (sel == 3'b001) ? path1_out :
                      (sel == 3'b010) ? path2_out : 8'd0;

endmodule

// Submodule for handling each data path
module mux_path (
    input clk,
    input [2:0] sel,
    input [7:0] data,
    output reg [7:0] path_out,
    input [2:0] sel_code // Path selection code
);

    always @(posedge clk) begin
        if (sel == sel_code) begin
            path_out <= data; // Pass data if selection matches
        end else begin
            path_out <= path_out; // Retain previous value
        end
    end

endmodule
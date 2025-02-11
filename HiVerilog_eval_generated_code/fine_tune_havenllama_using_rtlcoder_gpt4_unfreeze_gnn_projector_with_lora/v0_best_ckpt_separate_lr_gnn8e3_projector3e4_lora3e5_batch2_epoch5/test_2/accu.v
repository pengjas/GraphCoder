module accu (
    input               clk,
    input               rst_n,
    input       [7:0]   data_in,
    input               valid_in,

    output              valid_out,
    output      [9:0]   data_out
);


reg [2:0]   u_counter;

always @(posedge clk) begin
    if (!rst_n) begin
        u_counter <= 0;
    end else if (valid_in) begin
        u_counter <= u_counter + 1;
    end
end


reg [9:0]   u_data_accumulator;

always @(posedge clk) begin
    if (!rst_n) begin
        u_data_accumulator <= 0;
    end else if (valid_in) begin
        u_data_accumulator <= u_data_accumulator + data_in;
    end
end


reg                 u_valid_output;

always @(posedge clk) begin
    if (!rst_n) begin
        u_valid_output <= 0;
    end else if (u_counter == 4) begin
        u_valid_output <= 1;
    end
end

assign data_out = u_data_accumulator;
assign valid_out = u_valid_output;

endmodule
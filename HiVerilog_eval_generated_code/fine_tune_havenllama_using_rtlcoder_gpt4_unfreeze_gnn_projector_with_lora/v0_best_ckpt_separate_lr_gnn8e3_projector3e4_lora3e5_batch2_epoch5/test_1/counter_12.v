module counter_12 (
    input rst_n,
    input clk,
    input valid_count,
    output reg [3:0] out
);

    reg [3:0] counter;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter <= 4'b0000;
        end else if (valid_count) begin
            if (counter == 4'b1111) begin
                counter <= 4'b0000;
            end else begin
                counter <= counter + 1;
            end
        end
    end

    always @(posedge clk) begin
        if (valid_count) begin
            out <= counter;
        end
    end

endmodule

module
module output_register (
    input rst_n,
    input valid_count,
    input [3:0] counter,
    output reg [3:0] out
);

    always @(posedge rst_n or negedge valid_count) begin
        if (!rst_n) begin
            out <= 4'b0000;
        end else if (valid_count) begin
            out <= counter;
        end
    end

endmodule

module counter_12 (
    input rst_n,
    input clk,
    input valid_count,
    output reg [3:0] out
);

    counter_logic u_counter_logic (
        .rst_n(rst_n),
        .clk(clk),
        .valid_count(valid_count),
        .out(out)
    );

    output_register u_output_register (
        .rst_n(rst_n),
        .valid_count(valid_count),
        .counter(out),
        .out(out)
    );

endmodule
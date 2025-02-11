module multi_16bit(
    clk, 
    rst_n,
    start, 
    ain, 
    bin, 
    yout, 
    done
);
reg [3:0] shift_count;
reg done_flag;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        shift_count <= 4'd0;
        done_flag <= 1'b0;
    end else begin
        if (start) begin
            shift_count <= 4'd0;
            done_flag <= 1'b0;
        end else begin
            if (shift_count == 4'd15) begin
                shift_count <= 4'd0;
                done_flag <= 1'b1;
            end else begin
                shift_count <= shift_count + 4'd1;
                done_flag <= 1'b0;
            end
        end
    end
end


reg [15:0] a_reg;
reg [15:0] b_reg;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        a_reg <= 16'd0;
        b_reg <= 16'd0;
    end else begin
        if (start) begin
            a_reg <= ain;
            b_reg <= bin;
        end
    end
end


reg [31:0] product;
reg [15:0] multiplier;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        product <= 32'd0;
        multiplier <= 16'd0;
    end else begin
        if (shift_count == 4'd0) begin
            multiplier <= b_reg;
        end else begin
            multiplier <= {16'd0, multiplier[15:1]};
        end
        if (shift_count == 4'd15) begin
            product <= product + (multiplier * a_reg);
        end
    end
end


assign yout = product[31:2];
assign done = done_flag;

endmodule